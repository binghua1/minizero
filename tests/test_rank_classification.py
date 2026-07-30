import unittest

import torch

from minizero.network.py.alphazero_network import AlphaZeroNetwork


class RankClassificationTest(unittest.TestCase):
    def setUp(self):
        self.network = AlphaZeroNetwork(
            game_name="blokus10",
            num_input_channels=8,
            input_channel_height=10,
            input_channel_width=10,
            num_hidden_channels=16,
            hidden_channel_height=10,
            hidden_channel_width=10,
            num_blocks=1,
            action_size=101,
            num_value_hidden_channels=32,
            discrete_value_size=1,
            num_players=4,
        )

    def test_rank_outputs_are_probabilities_and_expected_utilities(self):
        output = self.network(torch.zeros(2, 8, 10, 10))

        self.assertEqual(output["rank_logit"].shape, (2, 4, 4))
        self.assertEqual(output["rank_probability"].shape, (2, 4, 4))
        self.assertEqual(output["rank"].shape, (2, 4))
        torch.testing.assert_close(
            output["rank_probability"].sum(dim=2),
            torch.ones(2, 4),
        )

        utilities = torch.tensor([1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0])
        expected = (output["rank_probability"] * utilities.view(1, 1, 4)).sum(dim=2)
        torch.testing.assert_close(output["rank"], expected)

    def test_network_is_torchscript_compatible(self):
        scripted = torch.jit.script(self.network)
        output = scripted(torch.zeros(1, 8, 10, 10))
        self.assertEqual(output["rank"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
