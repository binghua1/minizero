import unittest

try:
    import torch
    from minizero.network.py.alphazero_network import AlphaZeroNetwork
except ModuleNotFoundError:
    torch = None
    AlphaZeroNetwork = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BehaviorConditioningTest(unittest.TestCase):
    def create_model(self):
        return AlphaZeroNetwork(
            "tictacmo", 6, 3, 5, 16, 3, 5, 1, 15, 32, 1,
            num_players=3,
            use_rank_head=True,
            use_behavior_conditioning=True,
            behavior_history_length=4,
            behavior_embedding_dim=8,
            behavior_history_dropout=0.1,
        )

    def test_forward_and_torchscript_contract(self):
        model = self.create_model().eval()
        state = torch.randn(2, 6, 3, 5)
        history = torch.tensor([
            [[15, 15, 0, 3], [15, 1, 4, 7], [15, 2, 5, 8]],
            [[15, 15, 6, 9], [15, 0, 3, 10], [15, 1, 4, 11]],
        ])
        to_play = torch.tensor([0, 2])

        output = model(state, history, to_play)
        self.assertEqual(output["policy"].shape, (2, 15))
        self.assertEqual(output["value"].shape, (2, 3))
        self.assertEqual(output["rank"].shape, (2, 3))
        self.assertEqual(model.get_behavior_history_length(), 4)
        self.assertTrue(model.get_use_rank_head())

        scripted = torch.jit.script(model)
        scripted_output = scripted(state, history, to_play)
        self.assertEqual(scripted_output["value"].shape, (2, 3))
        self.assertEqual(scripted_output["rank"].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
