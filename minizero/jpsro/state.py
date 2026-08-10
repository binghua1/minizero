"""Persistent policy registry and empirical payoff table for JPSRO."""

import dataclasses
import itertools
import json
import math
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Policy:
    policy_id: str
    model_path: str
    players: tuple
    generation: int = 0
    metadata: dict = dataclasses.field(default_factory=dict)

    def to_dict(self):
        value = dataclasses.asdict(self)
        value["players"] = list(self.players)
        return value


class PayoffTable:
    """Sparse online mean/variance estimates indexed by ordered joint profile."""

    def __init__(self, num_players, entries=None):
        self.num_players = num_players
        self.entries = entries or {}

    @staticmethod
    def key(profile):
        return "\t".join(profile)

    def add(self, profile, returns, sample_id=None):
        if len(profile) != self.num_players or len(returns) != self.num_players:
            raise ValueError("profile and return vector must match num_players")
        key = self.key(profile)
        entry = self.entries.setdefault(key, {
            "profile": list(profile), "count": 0,
            "mean": [0.0] * self.num_players, "m2": [0.0] * self.num_players,
            "sample_ids": [],
        })
        entry.setdefault("sample_ids", [])
        if sample_id is not None and sample_id in entry["sample_ids"]:
            return False
        if sample_id is not None:
            entry["sample_ids"].append(sample_id)
        entry["count"] += 1
        count = entry["count"]
        for player, value in enumerate(returns):
            delta = float(value) - entry["mean"][player]
            entry["mean"][player] += delta / count
            entry["m2"][player] += delta * (float(value) - entry["mean"][player])
        return True

    def get(self, profile):
        return self.entries.get(self.key(profile))

    def stderr(self, profile):
        entry = self.get(profile)
        if not entry or entry["count"] < 2:
            return [math.inf] * self.num_players
        return [math.sqrt(value / (entry["count"] - 1) / entry["count"])
                for value in entry["m2"]]

    def missing(self, policy_sets, min_games=1):
        result = []
        for profile in itertools.product(*policy_sets):
            entry = self.get(profile)
            if not entry or entry["count"] < min_games:
                result.append(profile)
        return result

    def indexed_means(self, policy_sets, min_games=1):
        result = {}
        for index_profile in itertools.product(*(range(len(items)) for items in policy_sets)):
            profile = tuple(policy_sets[p][a] for p, a in enumerate(index_profile))
            entry = self.get(profile)
            if not entry or entry["count"] < min_games:
                raise ValueError(f"payoff is missing for profile {profile}")
            result[index_profile] = tuple(entry["mean"])
        return result

    def to_dict(self):
        return {"num_players": self.num_players, "entries": self.entries}

    @classmethod
    def from_dict(cls, data):
        return cls(int(data["num_players"]), data.get("entries", {}))


class JPSROState:
    """Filesystem-backed JPSRO run state."""

    def __init__(self, root, config, policies=None, payoff_table=None):
        self.root = Path(root).resolve()
        self.config = config
        self.policies = policies or {}
        self.payoffs = payoff_table or PayoffTable(config["num_players"])

    @property
    def num_players(self):
        return int(self.config["num_players"])

    @property
    def shared_pool(self):
        return bool(self.config.get("shared_pool", False))

    @classmethod
    def create(cls, root, num_players, shared_pool=False, utility_min=-1.0,
               utility_max=1.0):
        if not 2 <= num_players <= 6:
            raise ValueError("MiniZero JPSRO supports 2 to 6 players")
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        state = cls(root, {
            "version": 1,
            "num_players": num_players,
            "shared_pool": bool(shared_pool),
            "utility_min": float(utility_min),
            "utility_max": float(utility_max),
            "generation": 0,
        })
        state.save()
        return state

    @classmethod
    def load(cls, root):
        root = Path(root).resolve()
        config = json.loads((root / "state.json").read_text())
        policies_data = json.loads((root / "policies.json").read_text())
        policies = {
            item["policy_id"]: Policy(
                item["policy_id"], item["model_path"], tuple(item["players"]),
                int(item.get("generation", 0)), item.get("metadata", {}),
            ) for item in policies_data["policies"]
        }
        payoff_data = json.loads((root / "payoffs.json").read_text())
        return cls(root, config, policies, PayoffTable.from_dict(payoff_data))

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_json("state.json", self.config)
        self._write_json("policies.json", {
            "policies": [item.to_dict() for item in self.policies.values()]
        })
        self._write_json("payoffs.json", self.payoffs.to_dict())

    def _write_json(self, name, value):
        path = self.root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    def add_policy(self, policy_id, model_path, players=None, generation=None,
                   metadata=None):
        if not policy_id or any(char in policy_id for char in "\t, "):
            raise ValueError("policy_id must be non-empty and contain no spaces, tabs, or commas")
        if policy_id in self.policies:
            raise ValueError(f"policy already exists: {policy_id}")
        model_path = str(Path(model_path).resolve())
        if not Path(model_path).is_file():
            raise ValueError(f"model file does not exist: {model_path}")
        if any(char in model_path for char in " ,\t\r\n"):
            raise ValueError("model paths used by JPSRO cannot contain spaces, commas, or tabs")
        if players is None:
            players = tuple(range(self.num_players)) if self.shared_pool else ()
        players = tuple(sorted(set(int(player) for player in players)))
        if not players or any(player < 0 or player >= self.num_players for player in players):
            raise ValueError("policy must belong to at least one valid player")
        policy_generation = self.config["generation"] if generation is None else int(generation)
        self.config["generation"] = max(int(self.config["generation"]), policy_generation)
        self.policies[policy_id] = Policy(
            policy_id, model_path, players,
            policy_generation,
            metadata or {},
        )
        self.save()

    def policy_sets(self):
        return [
            [policy_id for policy_id, policy in self.policies.items() if player in policy.players]
            for player in range(self.num_players)
        ]

    def model_path(self, policy_id):
        return self.policies[policy_id].model_path

    def write_meta_strategy(self, result):
        distribution = [
            {"profile": [self.policy_sets()[p][action] for p, action in enumerate(profile)],
             "probability": probability}
            for profile, probability in sorted(result["distribution"].items())
        ]
        output = {key: value for key, value in result.items() if key != "distribution"}
        output["distribution"] = distribution
        output["marginals"] = self.marginals(distribution)
        self._write_json("meta_strategy.json", output)
        return output

    def load_meta_strategy(self):
        return json.loads((self.root / "meta_strategy.json").read_text())

    def marginals(self, distribution):
        result = [dict() for _ in range(self.num_players)]
        for item in distribution:
            for player, policy_id in enumerate(item["profile"]):
                result[player][policy_id] = result[player].get(policy_id, 0.0) + item["probability"]
        return result
