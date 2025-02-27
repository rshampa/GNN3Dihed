import torch
import torch.nn as nn

device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")  #shampa

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
        attention_weights = torch.nn.functional.softmax(dot_product, dim = 0)  #shampa: added dim=0 to avoid warning

        # Getting the value vectors
        value_vectors = self.value_matrix(x)

        # Getting the weighted sum of the value vectors
        weighted_sum = torch.matmul(attention_weights, value_vectors)

        return weighted_sum
class DMPNNLayer(nn.Module):

    def __init__(self, node_vector_size, edge_vector_size, activation_function = torch.nn.GELU):
        super().__init__()

        self.node_vector_size = node_vector_size
        self.edge_vector_size = edge_vector_size

        # Making a neural network for message passing (single layer)
        self.message_generation_network = torch.nn.Sequential(
            torch.nn.Linear(2 * node_vector_size + edge_vector_size, int((2 * node_vector_size + edge_vector_size) / 2)),
            activation_function(),
            torch.nn.Linear(int((2 * node_vector_size + edge_vector_size) / 2), node_vector_size),
            activation_function(),
        )

        # Making a neural network for combining the messages with the node vectors
        self.combination_network = torch.nn.Sequential(
            torch.nn.Linear(node_vector_size * 2, int((3 * node_vector_size) / 2)),
            activation_function(),
            torch.nn.Linear(int((3 * node_vector_size) / 2), node_vector_size),
            activation_function(),
        )

    def forward(self, x):
        node_vectors, edge_vectors, edge_indices = x

        # Printing out the shapes of the node vectors and edge vectors
        # print("Node vectors shape:", node_vectors.shape, "self.node_vector_size:", self.node_vector_size)
        assert node_vectors.shape[1] == self.node_vector_size
        # print("Edge vectors shape:", edge_vectors.shape, "self.edge_vector_size:", self.edge_vector_size)
        assert edge_vectors.shape[1] == self.edge_vector_size

        # Generating messages
       #messages_boxes = torch.zeros_like(node_vectors, device = "cuda:0")   #shampa
        messages_boxes = torch.zeros_like(node_vectors).to(device)           #shampa

        # print("node vector shape:", node_vectors.shape)
        # print("bond vector shape:", edge_vectors.shape)
        # print("")
       #neighbor_edge_concatenated_vectors = torch.zeros([2 * edge_vectors.size()[0], 2 * node_vectors.shape[1] + edge_vectors.shape[1]], device = "cuda:0")  #shampa
        neighbor_edge_concatenated_vectors = torch.zeros([2 * edge_vectors.size()[0], 2 * node_vectors.shape[1] + edge_vectors.shape[1]]).to(device)          #shampa
        # print("NEighbor edge concatenated vectors shape:", neighbor_edge_concatenated_vectors.shape)
       #node_indices = torch.zeros([2 * edge_vectors.size()[0]], dtype = torch.long, device = "cuda:0")  #shampa
        node_indices = torch.zeros([2 * edge_vectors.size()[0]], dtype = torch.long).to(device)          #shampa

        # Temporarily hardcoded GPU device instructions in many areas.

        for i, edge_index in enumerate(edge_indices):
            atom_1 = edge_index[0]
            atom_2 = edge_index[1]

            node_edge_pair_for_1 = torch.cat([node_vectors[atom_1], node_vectors[atom_2], edge_vectors[i]])
            node_edge_pair_for_2 = torch.cat([node_vectors[atom_2], node_vectors[atom_1], edge_vectors[i]])

            # Adding this to neighbor_edge_concatenated_vectors
            neighbor_edge_concatenated_vectors[2 * i] = node_edge_pair_for_1
            neighbor_edge_concatenated_vectors[2 * i + 1] = node_edge_pair_for_2

            # Adding the node indices to the node_indices tensor
            node_indices[2 * i] = atom_1
            node_indices[2 * i + 1] = atom_2

        # Passing the concatenated vectors through a neural network
        # to get the messages
        messages = self.message_generation_network(neighbor_edge_concatenated_vectors)

        # Aggregating the messages via sum
        for i, node_index in enumerate(node_indices):
            messages_boxes[node_index] += messages[i]

        # Message boxes are then concatenated with the corresponding node vectors
        node_message_pairs = torch.cat([node_vectors, messages_boxes], dim = 1)

        # Applying combination network to get new set of node vectors
        new_node_vectors = self.combination_network(node_message_pairs)

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

            


            