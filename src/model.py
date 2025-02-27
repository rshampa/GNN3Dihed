import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Non-variational vanilla autoencoder
class Autoencoder(torch.nn.Module):
    def __init__(self, original_space_dimensionality, latent_space_dimensionality, reduction_factor = 0.3, activation_function = torch.nn.GELU):
        super(Autoencoder, self).__init__()
        self.original_space_dimensionality = original_space_dimensionality
        self.latent_space_dimensionality = latent_space_dimensionality

        dimension_difference = original_space_dimensionality - latent_space_dimensionality
        reduction_term = int(reduction_factor * dimension_difference)

        layer_dimensions = [original_space_dimensionality]

        while True:
            if layer_dimensions[-1] - reduction_term > latent_space_dimensionality:
                layer_dimensions.append(layer_dimensions[-1] - reduction_term)
            else:
                layer_dimensions.append(latent_space_dimensionality)
                break

        # Making a sequential model for the encoder according to the layer dimensions
        self.encoder = torch.nn.Sequential()
        for i in range(1, len(layer_dimensions)):
            self.encoder.add_module("encoder_layer_" + str(i), torch.nn.Linear(layer_dimensions[i - 1], layer_dimensions[i]))
            self.encoder.add_module("encoder_activation_" + str(i), activation_function())

        # Making a sequential model for the decoder according to the layer dimensions
        self.decoder = torch.nn.Sequential()
        for i in range(len(layer_dimensions) - 1, 0, -1):
            self.decoder.add_module("decoder_layer_" + str(i), torch.nn.Linear(layer_dimensions[i], layer_dimensions[i - 1]))
            self.decoder.add_module("decoder_activation_" + str(i), activation_function())

    def forward(self, x):
        # Forward pass through the
        # encoder and decoder
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class AtomAutoencoder(torch.nn.Module):

    def __init__(self, input_size, latent_size):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, latent_size)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_size, input_size)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class BondAutoencoder(torch.nn.Module):

    def __init__(self, input_size, latent_size):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, latent_size)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_size, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, input_size)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


"""
The aggregation and combinations will all be simple matrix multiplications
followed by normalization via activation functions.
"""

class AggregateSelfAttention(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.query_matrix = torch.nn.Linear(embedding_size, embedding_size)
        self.key_matrix = torch.nn.Linear(embedding_size, embedding_size)
        self.value_matrix = torch.nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        query_vectors = self.query_matrix(x)
        key_vectors = self.key_matrix(x)

        # Calculating the dot product of the query and key vectors by torch dotproduct function along a dimension
        dot_product = torch.sum(torch.mul(query_vectors, key_vectors), dim = 1)

        # print("Dot product shape:", dot_product.shape)
        # print("Dot product:", dot_product)

        # Normalizing the dot product
        attention_weights = torch.nn.functional.softmax(dot_product, dim = 0)

        # Getting the value vectors
        value_vectors = self.value_matrix(x)

        # Getting the weighted sum of the value vectors
        weighted_sum = torch.matmul(attention_weights, value_vectors)

        return weighted_sum


class DMPNNLayer(nn.Module):
    """
    A simple Message Passing Neural Network layer (single direction).
    For an edge (i, j), we only compute the message from i -> j.
    """

    def __init__(
        self,
        node_vector_size,
        edge_vector_size,
        activation_function=nn.GELU,
        dropout_p=0.0,
        use_residual=True,
    ):
        super().__init__()
        self.node_vector_size = node_vector_size
        self.edge_vector_size = edge_vector_size
        self.use_residual = use_residual

        hidden_message_size = (2 * node_vector_size + edge_vector_size) // 2
        hidden_combination_size = (3 * node_vector_size) // 2

        # Network that generates messages for i->j
        self.message_generation_network = nn.Sequential(
            nn.Linear(2 * node_vector_size + edge_vector_size, hidden_message_size),
            activation_function(),
            nn.Linear(hidden_message_size, node_vector_size),
            activation_function(),
        )

        # Network that combines node's own embedding with the aggregated message
        self.combination_network = nn.Sequential(
            nn.Linear(node_vector_size * 2, hidden_combination_size),
            activation_function(),
            nn.Linear(hidden_combination_size, node_vector_size),
            activation_function(),
        )

        # Optional dropout
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: A tuple of (node_vectors, edge_vectors, edge_indices)
               where:
                 - node_vectors: shape [num_nodes, node_vector_size]
                 - edge_vectors: shape [num_edges, edge_vector_size]
                 - edge_indices: shape [num_edges, 2], each entry is (i, j)
                               indicating an edge from node i to node j
        Returns:
            new_node_vectors: shape [num_nodes, node_vector_size]
        """
        node_vectors, edge_vectors, edge_indices = x
        num_nodes = node_vectors.size(0)

        # Basic checks
        assert node_vectors.shape[1] == self.node_vector_size
        # print("L:", edge_vectors.shape)
        assert edge_vectors.shape[1] == self.edge_vector_size

        # -----------------------------------------------------
        # 1) Build the single-direction (i->j) concatenated vectors
        # -----------------------------------------------------
        node_i = node_vectors[edge_indices[:, 0]]  # shape: [E, node_dim]
        node_j = node_vectors[edge_indices[:, 1]]  # shape: [E, node_dim]

        # Concatenate: [node_i, node_j, edge_ij]
        # shape: [E, 2*node_dim + edge_dim]
        msg_input = torch.cat([node_i, node_j, edge_vectors], dim=1)

        # 2) Pass through message_generation_network: [E, node_dim]
        messages = self.message_generation_network(msg_input)
        messages = self.dropout(messages)

        # 3) Accumulate messages into their receivers (node j)
        messages_boxes = torch.zeros_like(node_vectors)  # [N, node_dim]

        receiver_indices = edge_indices[:, 1]  # [E]
        messages_boxes.index_add_(0, receiver_indices, messages)

        # 4) Combine node's own vector with aggregated messages
        node_message_pairs = torch.cat([node_vectors, messages_boxes], dim=1)  # [N, 2*node_dim]
        combined = self.combination_network(node_message_pairs)
        combined = self.dropout(combined)

        # 5) Optional residual (skip) connection
        new_node_vectors = node_vectors + combined if self.use_residual else combined

        return new_node_vectors


class GNN3D(nn.Module):

    def __init__(self, atomic_vector_size, bond_vector_size, number_of_molecular_features, number_of_targets, number_of_message_passes = 3):
        super().__init__()
        # It is assumed that the size of angles is 1.

        self.atom_bond_operator = DMPNNLayer(atomic_vector_size, bond_vector_size)
        self.bond_angle_operator = DMPNNLayer(bond_vector_size, 1)
        self.angle_dihedral_operator = DMPNNLayer(1, 1)
        #self.readout_aggregator = AggregateSelfAttention(atomic_vector_size)
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(atomic_vector_size + number_of_molecular_features, 30),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(30, number_of_targets)
            )

        self.number_of_message_passes = number_of_message_passes


    def forward(self, x):
        atomic_features, bond_features, angle_features, dihedral_features, global_molecular_features, bond_indices, angle_indices, dihedral_indices = x

        for _ in range(self.number_of_message_passes):
            #if dihedral_features.size()[0] != 0:
            #    angle_features = self.angle_dihedral_operator([torch.reshape(angle_features, [-1, 1]), torch.reshape(dihedral_features, [-1, 1]), dihedral_indices])

            bond_features = self.bond_angle_operator([bond_features, torch.reshape(angle_features, [-1, 1]), angle_indices])
            atomic_features = self.atom_bond_operator([atomic_features, bond_features, bond_indices])

        # Summing the atomic features
        readout_vector = torch.sum(atomic_features, dim = 0)
        #readout_vector = self.readout_aggregator(atomic_features)
        # print("readout_vector shape:", readout_vector.shape)
        # print("readout_vector:", readout_vector)

        # Concatenating readout_vector with global molecular features
        readout_vector = torch.cat([readout_vector, global_molecular_features])

        return self.readout(readout_vector)


class GNN3Dihed(nn.Module):

    def __init__(self, atomic_vector_size, bond_vector_size, number_of_molecular_features, number_of_targets, number_of_message_passes = 3):
        super().__init__()
        # It is assumed that the size of angles is 1.

        self.atom_bond_operator = DMPNNLayer(atomic_vector_size, bond_vector_size)
        self.bond_angle_operator = DMPNNLayer(bond_vector_size, 1)
        self.angle_dihedral_operator = DMPNNLayer(1, 1)
        self.readout_aggregator = AggregateSelfAttention(atomic_vector_size)
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(atomic_vector_size + number_of_molecular_features, 30),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(30, number_of_targets)
            )

        self.number_of_message_passes = number_of_message_passes


    def forward(self, x):
        atomic_features, bond_features, angle_features, dihedral_features, global_molecular_features, bond_indices, angle_indices, dihedral_indices = x

        for _ in range(self.number_of_message_passes):
            if dihedral_features.size()[0] != 0:
                angle_features = self.angle_dihedral_operator([torch.reshape(angle_features, [-1, 1]), torch.reshape(dihedral_features, [-1, 1]), dihedral_indices])

            bond_features = self.bond_angle_operator([bond_features, torch.reshape(angle_features, [-1, 1]), angle_indices])
            atomic_features = self.atom_bond_operator([atomic_features, bond_features, bond_indices])

        # Summing the atomic features
        #readout_vector = torch.sum(atomic_features, dim = 0)
        readout_vector = self.readout_aggregator(atomic_features)
        # print("readout_vector shape:", readout_vector.shape)
        # print("readout_vector:", readout_vector)

        # Concatenating readout_vector with global molecular features
        readout_vector = torch.cat([readout_vector, global_molecular_features])

        return self.readout(readout_vector)


class GNN3DAtnON(nn.Module):

    def __init__(self, atomic_vector_size, bond_vector_size, number_of_molecular_features, number_of_targets, number_of_message_passes = 3):
        super().__init__()
        # It is assumed that the size of angles is 1.

        self.atom_bond_operator = DMPNNLayer(atomic_vector_size, bond_vector_size)
        self.bond_angle_operator = DMPNNLayer(bond_vector_size, 1)
        self.angle_dihedral_operator = DMPNNLayer(1, 1)
        self.readout_aggregator = AggregateSelfAttention(atomic_vector_size)
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(atomic_vector_size + number_of_molecular_features, 30),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(30, number_of_targets)
            )

        self.number_of_message_passes = number_of_message_passes


    def forward(self, x):
        atomic_features, bond_features, angle_features, dihedral_features, global_molecular_features, bond_indices, angle_indices, dihedral_indices = x

        for _ in range(self.number_of_message_passes):
            #if dihedral_features.size()[0] != 0:
            #    angle_features = self.angle_dihedral_operator([torch.reshape(angle_features, [-1, 1]), torch.reshape(dihedral_features, [-1, 1]), dihedral_indices])

            bond_features = self.bond_angle_operator([bond_features, torch.reshape(angle_features, [-1, 1]), angle_indices])
            atomic_features = self.atom_bond_operator([atomic_features, bond_features, bond_indices])

        # Summing the atomic features
        #readout_vector = torch.sum(atomic_features, dim = 0)
        readout_vector = self.readout_aggregator(atomic_features)
        # print("readout_vector shape:", readout_vector.shape)
        # print("readout_vector:", readout_vector)

        # Concatenating readout_vector with global molecular features
        readout_vector = torch.cat([readout_vector, global_molecular_features])

        return self.readout(readout_vector)


class GNN3DAtnOFF(nn.Module):

    def __init__(self, atomic_vector_size, bond_vector_size, number_of_molecular_features, number_of_targets, number_of_message_passes = 3):
        super().__init__()
        # It is assumed that the size of angles is 1.

        self.atom_bond_operator = DMPNNLayer(atomic_vector_size, bond_vector_size)
        self.bond_angle_operator = DMPNNLayer(bond_vector_size, 1)
        self.angle_dihedral_operator = DMPNNLayer(1, 1)
        #self.readout_aggregator = AggregateSelfAttention(atomic_vector_size)
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(atomic_vector_size + number_of_molecular_features, 30),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(30, number_of_targets)
            )

        self.number_of_message_passes = number_of_message_passes


    def forward(self, x):
        atomic_features, bond_features, angle_features, dihedral_features, global_molecular_features, bond_indices, angle_indices, dihedral_indices = x

        for _ in range(self.number_of_message_passes):
            if dihedral_features.size()[0] != 0:
                angle_features = self.angle_dihedral_operator([torch.reshape(angle_features, [-1, 1]), torch.reshape(dihedral_features, [-1, 1]), dihedral_indices])

            bond_features = self.bond_angle_operator([bond_features, torch.reshape(angle_features, [-1, 1]), angle_indices])
            atomic_features = self.atom_bond_operator([atomic_features, bond_features, bond_indices])

        # Summing the atomic features
        readout_vector = torch.sum(atomic_features, dim = 0)
        #readout_vector = self.readout_aggregator(atomic_features)
        # print("readout_vector shape:", readout_vector.shape)
        # print("readout_vector:", readout_vector)

        # Concatenating readout_vector with global molecular features
        readout_vector = torch.cat([readout_vector, global_molecular_features])

        return self.readout(readout_vector)


class GNN3DConfig:

    def __init__(self, atomic_vector_size, bond_vector_size,
                 number_of_molecular_features,
                 number_of_message_passes = 3, use_bond_angles = True, use_dihedral_angles = True,
                 use_attention_based_readout = True):

        self.atomic_vector_size = atomic_vector_size
        self.bond_vector_size = bond_vector_size
        self.number_of_molecular_features = number_of_molecular_features
        self.number_of_message_passes = number_of_message_passes
        self.use_bond_angles = use_bond_angles
        # Cannot use dihedral angles without bond angles
        self.use_dihedral_angles = use_dihedral_angles and use_bond_angles
        self.use_attention_based_readout = use_attention_based_readout


class GNN3DLayer(nn.Module):

    def __init__(self, config : GNN3DConfig):
        super().__init__()
        self.config = config


        # Setting up components
        # It is assumed that the size of angles is 1.

        self.atom_bond_operator = DMPNNLayer(config.atomic_vector_size, config.bond_vector_size)
        if config.use_bond_angles:
            self.bond_angle_operator = DMPNNLayer(config.bond_vector_size, 1)
        if config.use_dihedral_angles:
            self.angle_dihedral_operator = DMPNNLayer(1, 1)

        self.readout_aggregator = AggregateSelfAttention(config.atomic_vector_size) if config.use_attention_based_readout else None

    def forward(self, x):
        atomic_features, bond_features, angle_features, dihedral_features, global_molecular_features, bond_indices, angle_indices, dihedral_indices = x

        for _ in range(self.config.number_of_message_passes):
            if self.config.use_dihedral_angles and dihedral_features.size()[0] != 0:
                angle_features = self.angle_dihedral_operator([torch.reshape(angle_features, [-1, 1]), torch.reshape(dihedral_features, [-1, 1]), dihedral_indices])

            if self.config.use_bond_angles:
                bond_features = self.bond_angle_operator([bond_features, torch.reshape(angle_features, [-1, 1]), angle_indices])
            atomic_features = self.atom_bond_operator([atomic_features, bond_features, bond_indices])

        # Summing the atomic features
        readout_vector = torch.sum(atomic_features, dim = 0) if self.readout_aggregator is None else self.readout_aggregator(atomic_features)

        # Concatenating readout_vector with global molecular features
        readout_vector = torch.cat([readout_vector, global_molecular_features])

        return readout_vector


class GNN3DClassifier(torch.nn.Module):

    def __init__(self, gnn_config, number_of_classes):
        super().__init__()
        self.gnn_config = gnn_config
        self.number_of_targets = number_of_classes
        self.gnn_layer = GNN3DLayer(gnn_config)
        self.classifier_head = torch.nn.Sequential(
            torch.nn.Linear(gnn_config.atomic_vector_size + gnn_config.number_of_molecular_features, number_of_classes),
            torch.nn.Softmax(dim=0),
        )

    def forward(self, x):
        readout_vector = self.gnn_layer(x)
        return self.classifier_head(readout_vector)


class GNN3DMultiTaskClassifier(torch.nn.Module):
    def __init__(self, gnn_config, number_of_classes_per_task):
        """
        A multi-task learning model based on a GNN, with separate classification heads
        for each task. Each task is treated as a separate binary or multi-class classification problem.

        Parameters:
        - gnn_config: Configuration for the GNN layer.
        - number_of_classes_per_task: Number of classes for each task, either a list or a single integer.
        """
        super().__init__()
        self.gnn_config = gnn_config
        self.number_of_classes_per_task = number_of_classes_per_task
        self.gnn_layer = GNN3DLayer(gnn_config)
        if isinstance(number_of_classes_per_task, int):
            number_of_classes_per_task = [number_of_classes_per_task]

        # Create a classifier head for each task (using Sigmoid for multi-label classification)
        self.classifier_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gnn_config.atomic_vector_size + gnn_config.number_of_molecular_features, num_classes),
                nn.Sigmoid()
            )
            for num_classes in number_of_classes_per_task
        ])
       #print(f"No. of Tasks: {len(self.classifier_heads)}")

    def forward(self, x):
        """
        Forward pass of the model.
        Returns the prediction for each task.

        Parameters:
        - x: Input data to the GNN, which is expected to be a tuple of features.

        Returns:
        - A tuple of task outputs, one per task.
        """
        readout_vector = self.gnn_layer(x)
        task_outputs = [classifier_head(readout_vector) for classifier_head in self.classifier_heads]

        return tuple(task_outputs)


class GNN2D(nn.Module):

    def __init__(self, atomic_vector_size, bond_vector_size, number_of_molecular_features, number_of_targets, number_of_message_passes = 3):
        super().__init__()

        self.atom_bond_operator = DMPNNLayer(atomic_vector_size, bond_vector_size)
        self.readout_aggregator = AggregateSelfAttention(atomic_vector_size)
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(atomic_vector_size + number_of_molecular_features, 30),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(30, number_of_targets)
            )

        self.number_of_message_passes = number_of_message_passes


    def forward(self, x):
        atomic_features, bond_features, global_molecular_features, bond_indices = x

        for _ in range(self.number_of_message_passes):
            atomic_features = self.atom_bond_operator([atomic_features, bond_features, bond_indices])

        readout_vector = self.readout_aggregator(atomic_features)
        readout_vector = torch.cat([readout_vector, global_molecular_features])

        return self.readout(readout_vector)


class GNNBondAngle(nn.Module):

    def __init__(self, atomic_vector_size, bond_vector_size, number_of_molecular_features, number_of_targets, number_of_message_passes = 3):
        super().__init__()

        self.atom_bond_operator = DMPNNLayer(atomic_vector_size, bond_vector_size)
        self.bond_angle_operator = DMPNNLayer(bond_vector_size, 1)
        self.angle_dihedral_operator = DMPNNLayer(1, 1)
        self.readout_aggregator = AggregateSelfAttention(bond_vector_size)
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(bond_vector_size + number_of_molecular_features, 30),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(30, number_of_targets)
            )

        self.number_of_message_passes = number_of_message_passes


    def forward(self, x):
        bond_features, angle_features, global_molecular_features, angle_indices = x

        for _ in range(self.number_of_message_passes):
            bond_features = self.bond_angle_operator([bond_features, torch.reshape(angle_features, [-1, 1]), angle_indices])

        readout_vector = self.readout_aggregator(bond_features)
        readout_vector = torch.cat([readout_vector, global_molecular_features])

        return self.readout(readout_vector)
