import time
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from data_utils import parse_PDB
from model_utils_onnx import ONNXFriendlyLigandFeatureExtractor, LigandMPNNEncoder, ONNXFriendlyDecoderStep

aa_dict = {
    0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F',
    5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L',
    10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R',
    15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y',
    20: 'X'
}


def test_ligand_feature_extractor():
    """Test the ONNX-friendly ligand feature extractor"""

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model
    feature_extractor = ONNXFriendlyLigandFeatureExtractor(
        edge_features=128,
        node_features=128,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=48,
        device=device,
        atom_context_num=16
    ).to(device)

    # Load example PDB with ligand
    pdb_path = "1A3N.pdb"
    feature_dict, backbone, other_atoms, CA_icodes, water_atoms = parse_PDB(
        input_path=pdb_path,
        device=device,
        chains=["A"],
        parse_all_atoms=True
    )

    # Extract required features
    coords = feature_dict['X']  # [L,4,3] - backbone coordinates
    ligand_coords = feature_dict['Y']  # [M,3] - ligand coordinates
    ligand_types = feature_dict['Y_t']  # [M] - ligand atom types
    ligand_mask = feature_dict['Y_m']  # [M] - ligand mask

    # Add batch dimension and reshape ligand features
    coords = coords.unsqueeze(0)  # [1,L,4,3]

    # Expand ligand features to match sequence length
    L = coords.shape[1]
    ligand_coords = ligand_coords.unsqueeze(0).expand(1, L, -1, 3)  # [1,L,M,3]
    ligand_types = ligand_types.unsqueeze(0).expand(1, L, -1)  # [1,L,M]
    ligand_mask = ligand_mask.unsqueeze(0).expand(1, L, -1)  # [1,L,M]

    print("\nInput shapes:")
    print(f"Coords: {coords.shape}")
    print(f"Ligand coords: {ligand_coords.shape}")
    print(f"Ligand types: {ligand_types.shape}")
    print(f"Ligand mask: {ligand_mask.shape}")

    # Run feature extraction
    with torch.no_grad():
        V, Y_nodes, Y_edges, E_idx = feature_extractor(
            coords,
            ligand_coords,
            ligand_types,
            ligand_mask
        )

def load_ligand_mpnn():
    """Load LigandMPNN with pretrained weights"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LigandMPNNEncoder(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        k_neighbors=48,
        atom_context_num=16,
        device=device
    ).to(device)

    checkpoint = torch.load(
        "model_params/ligandmpnn_v_32_030_25.pt",
        map_location=device
    )

    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}

    # Correct mapping of keys
    key_mapping = {
        'features.node_project_down': 'feature_extractor.node_project_down',  # Fixed this
        'features.embeddings': 'feature_extractor.embeddings',
        'features.edge_embedding': 'feature_extractor.edge_embedding',
        'features.norm_edges': 'feature_extractor.norm_edges',
        'features.type_linear': 'feature_extractor.type_linear',
        'features.norm_nodes': 'feature_extractor.norm_nodes',
        'features.y_nodes': 'feature_extractor.y_nodes',
        'features.y_edges': 'feature_extractor.y_edges',
        'features.norm_y_edges': 'feature_extractor.norm_y_edges',
        'features.norm_y_nodes': 'feature_extractor.norm_y_nodes',
    }

    for k, v in state_dict.items():
        # Skip decoder layers and output layers that we don't need for encoder
        if k.startswith(('decoder_layers', 'W_out', 'W_s')):
            continue

        # Map feature extractor keys
        found = False
        for old_key, new_key in key_mapping.items():
            if k.startswith(old_key):
                new_k = k.replace(old_key, new_key)
                new_state_dict[new_k] = v
                found = True
                break

        # Keep other encoder keys as is
        if not found and not k.startswith('features'):
            if k == 'W_e.weight' or k == 'W_e.bias':
                continue  # Skip W_e weights as they're not in our model
            new_state_dict[k] = v

    # Load weights
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print("\nMissing keys:", missing)
    print("\nUnexpected keys:", unexpected)

    return model


def test_and_export_ligand_mpnn():
    """Test and export the LigandMPNN encoder to ONNX"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model with weights
    model = load_ligand_mpnn()
    model.eval()

    # Create example inputs
    B, L = 1, 141  # Batch size and sequence length
    M = 43  # Number of ligand atoms

    coords = torch.randn(B, L, 4, 3, device=device)
    ligand_coords = torch.randn(B, L, M, 3, device=device)
    ligand_types = torch.randint(0, 120, (B, L, M), device=device)
    ligand_mask = torch.ones(B, L, M, device=device)

    print("\nInput shapes:")
    print(f"Coords: {coords.shape}")
    print(f"Ligand coords: {ligand_coords.shape}")
    print(f"Ligand types: {ligand_types.shape}")
    print(f"Ligand mask: {ligand_mask.shape}")

    # Test forward pass
    with torch.no_grad():
          h_V, h_E, E_idx = model(coords, ligand_coords, ligand_types, ligand_mask)

          # Calculate logits for first few positions
          num_positions = 5
          for pos in range(num_positions):
              # Get logits for this position
              pos_logits = h_V[0, pos]  # [hidden_dim]

              # Only consider the first 20 amino acids (exclude X)
              pos_logits = pos_logits[:20]  # Take only first 20 logits

              # Convert to probabilities
              probs = torch.softmax(pos_logits / 0.1, dim=-1)  # temperature=0.1

              # Get top 5 amino acids
              top_probs, top_indices = torch.topk(probs, k=5)

              print(f"\nPosition {pos+1}:")
              print("Top 5 amino acids (PyTorch):")
              for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                  print(f"{aa_dict[idx]}: {prob:.4f}")

    # Export to ONNX
    print("\nExporting to ONNX...")
    torch.onnx.export(
        model,
        (coords, ligand_coords, ligand_types, ligand_mask),
        "ligand_encoder.onnx",
        input_names=['coords', 'ligand_coords', 'ligand_types', 'ligand_mask'],
        output_names=['h_V', 'h_E', 'E_idx'],
        dynamic_axes={
            'coords': {0: 'batch', 1: 'sequence'},
            'ligand_coords': {0: 'batch', 1: 'sequence', 2: 'num_atoms'},
            'ligand_types': {0: 'batch', 1: 'sequence', 2: 'num_atoms'},
            'ligand_mask': {0: 'batch', 1: 'sequence', 2: 'num_atoms'},
            'h_V': {0: 'batch', 1: 'sequence'},
            'h_E': {0: 'batch', 1: 'sequence'},
            'E_idx': {0: 'batch', 1: 'sequence'}
        },
        opset_version=11,
        do_constant_folding=True
    )

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    import onnxruntime as ort
    ort_session = ort.InferenceSession("ligand_encoder.onnx")

    # Prepare inputs for ONNX
    ort_inputs = {
        'coords': coords.cpu().numpy(),
        'ligand_coords': ligand_coords.cpu().numpy(),
        'ligand_types': ligand_types.cpu().numpy(),
        'ligand_mask': ligand_mask.cpu().numpy()
    }

    # Run ONNX inference
    ort_outputs = ort_session.run(None, ort_inputs)

    # Compare outputs
    print("\nComparing PyTorch and ONNX outputs:")
    torch_outputs = [h_V, h_E, E_idx]
    for torch_out, onnx_out, name in zip(torch_outputs, ort_outputs, ['h_V', 'h_E', 'E_idx']):
        max_diff = np.abs(torch_out.cpu().numpy() - onnx_out).max()
        print(f"{name} max difference: {max_diff:.6f}")


def test_ligand_mpnn_sequences():
    """Test LigandMPNN encoder and compare logits for different proteins"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # List of test PDBs
    pdb_paths = ["1A3N.pdb"]

    # Load model
    model = load_ligand_mpnn()
    model.eval()

    # Load ONNX session
    try:
        print("\nLoading ONNX model...")
        ort_session = ort.InferenceSession("ligand_encoder.onnx")
        print("ONNX model loaded successfully")

        # Print expected input details
        print("\nONNX model expected inputs:")
        for i, input_info in enumerate(ort_session.get_inputs()):
            print(f"Input {i}: name={input_info.name}, type={input_info.type}, shape={input_info.shape}")
    except Exception as e:
        print(f"Error loading ONNX model: {str(e)}")
        return

    for pdb_path in pdb_paths:
        print(f"\n{'='*80}")
        print(f"Processing {pdb_path}")
        print(f"{'='*80}")

        # Load PDB data
        try:
            feature_dict, backbone, other_atoms, CA_icodes, water_atoms = parse_PDB(
                input_path=pdb_path,
                device=device,
                chains=["A"],
                parse_all_atoms=True
            )
        except Exception as e:
            print(f"Error loading {pdb_path}: {str(e)}")
            continue

        # Prepare inputs
        coords = feature_dict['X'].unsqueeze(0)  # [1,L,4,3]
        ligand_coords = feature_dict['Y'].unsqueeze(0).unsqueeze(0).expand(-1, coords.shape[1], -1, -1)
        ligand_types = feature_dict['Y_t'].long().unsqueeze(0).unsqueeze(0).expand(-1, coords.shape[1], -1)
        ligand_mask = feature_dict['Y_m'].unsqueeze(0).unsqueeze(0).expand(-1, coords.shape[1], -1)

        print(f"\nStructure details:")
        print(f"Protein length: {coords.shape[1]}")
        print(f"Number of ligand atoms: {ligand_coords.shape[2]}")

        # PyTorch inference
        print("\nPyTorch inference:")
        try:
            with torch.no_grad():
                h_V, h_E, E_idx = model(coords, ligand_coords, ligand_types, ligand_mask)
                print("PyTorch inference successful")
                print(f"h_V shape: {h_V.shape}")
                print(f"h_E shape: {h_E.shape}")
                print(f"E_idx shape: {E_idx.shape}")
        except Exception as e:
            print(f"Error during PyTorch inference: {str(e)}")
            continue

        # ONNX inference
        print("\nPreparing ONNX inputs...")
        try:
            ort_inputs = {
                'coords': coords.cpu().numpy().astype(np.float32),
                'ligand_coords': ligand_coords.cpu().numpy().astype(np.float32),
                'ligand_types': ligand_types.cpu().numpy().astype(np.int64),
                'ligand_mask': ligand_mask.cpu().numpy().astype(np.float32)
            }

            # Print input shapes and types
            print("\nONNX input details:")
            for name, value in ort_inputs.items():
                print(f"{name}: shape={value.shape}, dtype={value.dtype}")

            print("\nRunning ONNX inference...")
            onnx_outputs = ort_session.run(None, ort_inputs)
            onnx_h_V, onnx_h_E, onnx_E_idx = onnx_outputs
            print("ONNX inference successful")
            print(f"onnx_h_V shape: {onnx_h_V.shape}")
            print(f"onnx_h_E shape: {onnx_h_E.shape}")
            print(f"onnx_E_idx shape: {onnx_E_idx.shape}")

        except Exception as e:
            print(f"Error during ONNX inference: {str(e)}")
            continue

        # Compare outputs
        num_positions = 5
        print("\nComparing outputs position by position:")
        for pos in range(num_positions):
            try:
                # Get PyTorch logits and probabilities
                pt_logits = h_V[0, pos, :20].cpu()
                pt_probs = torch.softmax(pt_logits / 0.1, dim=-1)

                # Get ONNX logits and probabilities
                onnx_logits = torch.tensor(onnx_h_V[0, pos, :20], device='cpu')
                onnx_probs = torch.softmax(onnx_logits / 0.1, dim=-1)

                print(f"\nPosition {pos+1}:")
                print("Top 5 predictions:")
                print(f"{'AA':<4} {'PyTorch':>10} {'ONNX':>10} {'Diff':>10}")
                print("-" * 35)

                # Get top 5 for both
                pt_top_probs, pt_top_indices = torch.topk(pt_probs, k=5)
                onnx_top_probs, onnx_top_indices = torch.topk(onnx_probs, k=5)

                for i in range(5):
                    pt_idx = pt_top_indices[i].item()
                    onnx_idx = onnx_top_indices[i].item()
                    pt_p = pt_top_probs[i].item()
                    onnx_p = onnx_top_probs[i].item()
                    print(f"{aa_dict[pt_idx]:<4} {pt_p:>10.4f} {aa_dict[onnx_idx]:>4} {onnx_p:>10.4f}")

            except Exception as e:
                print(f"Error comparing position {pos}: {str(e)}")
                continue

        # Overall statistics
        try:
            print("\nOverall comparison:")
            h_V_diff = np.abs(h_V.cpu().numpy() - onnx_h_V).max()
            h_E_diff = np.abs(h_E.cpu().numpy() - onnx_h_E).max()
            E_idx_diff = np.abs(E_idx.cpu().numpy() - onnx_E_idx).max()

            print(f"Maximum differences:")
            print(f"h_V: {h_V_diff:.6f}")
            print(f"h_E: {h_E_diff:.6f}")
            print(f"E_idx: {E_idx_diff:.6f}")
        except Exception as e:
            print(f"Error calculating overall statistics: {str(e)}")



def export_ligand_mpnn_decoder():
    """Export decoder step to ONNX"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize decoder
    decoder = ONNXFriendlyDecoderStep(
        hidden_dim=128,
        vocab_size=21
    ).to(device)

    # Load weights from checkpoint
    checkpoint = torch.load("model_params/ligandmpnn_v_32_030_25.pt", map_location=device)

    # Map decoder weights
    decoder_state = {
        'W_s.weight': checkpoint['model_state_dict']['W_s.weight'],
        'W_out.weight': checkpoint['model_state_dict']['W_out.weight'],
        'W_out.bias': checkpoint['model_state_dict']['W_out.bias'],

        # Decoder layer weights
        'decoder_layer.W1.weight': checkpoint['model_state_dict']['decoder_layers.0.W1.weight'],
        'decoder_layer.W1.bias': checkpoint['model_state_dict']['decoder_layers.0.W1.bias'],
        'decoder_layer.W2.weight': checkpoint['model_state_dict']['decoder_layers.0.W2.weight'],
        'decoder_layer.W2.bias': checkpoint['model_state_dict']['decoder_layers.0.W2.bias'],
        'decoder_layer.W3.weight': checkpoint['model_state_dict']['decoder_layers.0.W3.weight'],
        'decoder_layer.W3.bias': checkpoint['model_state_dict']['decoder_layers.0.W3.bias'],

        # Layer norms
        'decoder_layer.norm1.weight': checkpoint['model_state_dict']['decoder_layers.0.norm1.weight'],
        'decoder_layer.norm1.bias': checkpoint['model_state_dict']['decoder_layers.0.norm1.bias'],
        'decoder_layer.norm2.weight': checkpoint['model_state_dict']['decoder_layers.0.norm2.weight'],
        'decoder_layer.norm2.bias': checkpoint['model_state_dict']['decoder_layers.0.norm2.bias'],

        # Dense feedforward weights
        'decoder_layer.dense.W_in.weight': checkpoint['model_state_dict']['decoder_layers.0.dense.W_in.weight'],
        'decoder_layer.dense.W_in.bias': checkpoint['model_state_dict']['decoder_layers.0.dense.W_in.bias'],
        'decoder_layer.dense.W_out.weight': checkpoint['model_state_dict']['decoder_layers.0.dense.W_out.weight'],
        'decoder_layer.dense.W_out.bias': checkpoint['model_state_dict']['decoder_layers.0.dense.W_out.bias'],
    }

    # Load weights
    decoder.load_state_dict(decoder_state)
    decoder.eval()

    # Create example inputs for export
    B, L = 1, 141  # Batch size and sequence length
    h_V = torch.randn(B, L, 128, device=device)
    h_E = torch.randn(B, L, 16, 128, device=device)
    E_idx = torch.randint(0, L, (B, L, 16), device=device)
    position = torch.tensor([0], device=device)
    temperature = torch.tensor([0.1], device=device)

    # Export to ONNX
    print("\nExporting decoder to ONNX...")
    torch.onnx.export(
        decoder,
        (h_V, h_E, E_idx, position, temperature),
        "ligand_decoder.onnx",
        input_names=['h_V', 'h_E', 'E_idx', 'position', 'temperature'],
        output_names=['logits'],
        dynamic_axes={
            'h_V': {0: 'batch', 1: 'sequence'},
            'h_E': {0: 'batch', 1: 'sequence'},
            'E_idx': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch'}
        },
        opset_version=11,
        do_constant_folding=True
    )

    # Verify ONNX model
    print("\nVerifying decoder ONNX model...")
    ort_session = ort.InferenceSession("ligand_decoder.onnx")

    # Prepare inputs
    ort_inputs = {
        'h_V': h_V.cpu().numpy(),
        'h_E': h_E.cpu().numpy(),
        'E_idx': E_idx.cpu().numpy(),
        'position': position.cpu().numpy(),
        'temperature': temperature.cpu().numpy()
    }

    # Compare outputs
    with torch.no_grad():
        pt_output = decoder(h_V, h_E, E_idx, position, temperature)
        onnx_output = ort_session.run(None, ort_inputs)[0]

        max_diff = np.abs(pt_output.cpu().numpy() - onnx_output).max()
        print(f"\nMaximum difference between PyTorch and ONNX outputs: {max_diff:.6f}")


def test_ligand_mpnn_full():
    """Test both encoder and decoder ONNX models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load ONNX sessions
    print("\nLoading ONNX models...")
    encoder_session = ort.InferenceSession("ligand_encoder.onnx")
    decoder_session = ort.InferenceSession("ligand_decoder.onnx")
    print("ONNX models loaded successfully")

    # Load PyTorch models
    print("\nLoading PyTorch models...")
    encoder = load_ligand_mpnn()
    encoder.eval()

    decoder = ONNXFriendlyDecoderStep(
        hidden_dim=128,
        vocab_size=21
    ).to(device)

    # Load decoder weights from checkpoint
    checkpoint = torch.load("model_params/ligandmpnn_v_32_030_25.pt", map_location=device)

    # Map decoder weights
    decoder_state = {
        'W_s.weight': checkpoint['model_state_dict']['W_s.weight'],
        'W_out.weight': checkpoint['model_state_dict']['W_out.weight'],
        'W_out.bias': checkpoint['model_state_dict']['W_out.bias'],

        # Decoder layer weights
        'decoder_layer.W1.weight': checkpoint['model_state_dict']['decoder_layers.0.W1.weight'],
        'decoder_layer.W1.bias': checkpoint['model_state_dict']['decoder_layers.0.W1.bias'],
        'decoder_layer.W2.weight': checkpoint['model_state_dict']['decoder_layers.0.W2.weight'],
        'decoder_layer.W2.bias': checkpoint['model_state_dict']['decoder_layers.0.W2.bias'],
        'decoder_layer.W3.weight': checkpoint['model_state_dict']['decoder_layers.0.W3.weight'],
        'decoder_layer.W3.bias': checkpoint['model_state_dict']['decoder_layers.0.W3.bias'],

        # Layer norms
        'decoder_layer.norm1.weight': checkpoint['model_state_dict']['decoder_layers.0.norm1.weight'],
        'decoder_layer.norm1.bias': checkpoint['model_state_dict']['decoder_layers.0.norm1.bias'],
        'decoder_layer.norm2.weight': checkpoint['model_state_dict']['decoder_layers.0.norm2.weight'],
        'decoder_layer.norm2.bias': checkpoint['model_state_dict']['decoder_layers.0.norm2.bias'],

        # Dense feedforward weights
        'decoder_layer.dense.W_in.weight': checkpoint['model_state_dict']['decoder_layers.0.dense.W_in.weight'],
        'decoder_layer.dense.W_in.bias': checkpoint['model_state_dict']['decoder_layers.0.dense.W_in.bias'],
        'decoder_layer.dense.W_out.weight': checkpoint['model_state_dict']['decoder_layers.0.dense.W_out.weight'],
        'decoder_layer.dense.W_out.bias': checkpoint['model_state_dict']['decoder_layers.0.dense.W_out.bias'],
    }

    decoder.load_state_dict(decoder_state)
    decoder.eval()
    print("PyTorch models loaded successfully")

    # Test PDBs
    pdb_paths = ["1A3N.pdb"]

    for pdb_path in pdb_paths:
        print(f"\n{'='*80}")
        print(f"Processing {pdb_path}")
        print(f"{'='*80}")

        # Load PDB data
        feature_dict, backbone, other_atoms, CA_icodes, water_atoms = parse_PDB(
            input_path=pdb_path,
            device=device,
            chains=["A"],
            parse_all_atoms=True
        )

        # Prepare inputs
        coords = feature_dict['X'].unsqueeze(0)  # [1,L,4,3]
        ligand_coords = feature_dict['Y'].unsqueeze(0).unsqueeze(0).expand(-1, coords.shape[1], -1, -1)
        ligand_types = feature_dict['Y_t'].long().unsqueeze(0).unsqueeze(0).expand(-1, coords.shape[1], -1)
        ligand_mask = feature_dict['Y_m'].unsqueeze(0).unsqueeze(0).expand(-1, coords.shape[1], -1)

        print(f"\nStructure details:")
        print(f"Protein length: {coords.shape[1]}")
        print(f"Number of ligand atoms: {ligand_coords.shape[2]}")

        # PyTorch full sequence generation
        print("\nGenerating sequence with PyTorch model...")
        with torch.no_grad():
            # Encoder
            start_time = time.time()
            pt_h_V, pt_h_E, pt_E_idx = encoder(coords, ligand_coords, ligand_types, ligand_mask)
            encoder_time = time.time() - start_time
            print(f"PyTorch encoder time: {encoder_time:.4f}s")

            # Generate sequence
            temperatures = [0.1, 0.5, 1.0]

            for temp in temperatures:
                print(f"\nTemperature: {temp}")
                sequence = []
                decoder_time = 0

                for pos in range(5):  # Generate first 5 positions
                    position = torch.tensor([pos], device=device)
                    temperature = torch.tensor([temp], device=device)

                    # Get logits for current position
                    start_time = time.time()
                    logits = decoder(
                        pt_h_V,
                        pt_h_E,
                        pt_E_idx,
                        position,
                        temperature
                    )
                    decoder_time += time.time() - start_time

                    # Convert to probabilities
                    probs = torch.softmax(logits[0, :20] / temp, dim=-1)

                    # Get top predictions
                    top_probs, top_indices = torch.topk(probs, k=5)

                    print(f"\nPosition {pos+1} (PyTorch):")
                    print(f"{'AA':<4} {'Prob':>10}")
                    print("-" * 15)
                    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                        print(f"{aa_dict[idx]:<4} {prob:>10.4f}")

                    # Sample from distribution
                    aa_idx = torch.multinomial(probs, 1)[0].item()
                    sequence.append(aa_dict[aa_idx])

                print(f"\nGenerated sequence: {''.join(sequence)}")
                print(f"PyTorch average decoder time per position: {decoder_time/5:.4f}s")

        # ONNX sequence generation
        print("\nGenerating sequence with ONNX models...")

        # Prepare encoder inputs
        ort_encoder_inputs = {
            'coords': coords.cpu().numpy().astype(np.float32),
            'ligand_coords': ligand_coords.cpu().numpy().astype(np.float32),
            'ligand_types': ligand_types.cpu().numpy().astype(np.int64),
            'ligand_mask': ligand_mask.cpu().numpy().astype(np.float32)
        }

        # Run encoder
        start_time = time.time()
        onnx_h_V, onnx_h_E, onnx_E_idx = encoder_session.run(None, ort_encoder_inputs)
        encoder_time = time.time() - start_time
        print(f"ONNX encoder time: {encoder_time:.4f}s")

        # Generate sequence with different temperatures
        for temp in temperatures:
            print(f"\nTemperature: {temp}")
            sequence = []
            decoder_time = 0

            for pos in range(5):
                # Prepare decoder inputs
                ort_decoder_inputs = {
                    'h_V': onnx_h_V,
                    'h_E': onnx_h_E,
                    'E_idx': onnx_E_idx,
                    'position': np.array([pos], dtype=np.int64),
                    'temperature': np.array([temp], dtype=np.float32)
                }

                # Run decoder
                start_time = time.time()
                onnx_logits = decoder_session.run(None, ort_decoder_inputs)[0]
                decoder_time += time.time() - start_time

                # Convert to probabilities
                probs = torch.softmax(torch.tensor(onnx_logits[0, :20]) / temp, dim=-1)

                # Get top predictions
                top_probs, top_indices = torch.topk(probs, k=5)

                print(f"\nPosition {pos+1} (ONNX):")
                print(f"{'AA':<4} {'Prob':>10}")
                print("-" * 15)
                for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                    print(f"{aa_dict[idx]:<4} {prob:>10.4f}")

                # Sample from distribution
                aa_idx = torch.multinomial(probs, 1)[0].item()
                sequence.append(aa_dict[aa_idx])

            print(f"\nGenerated sequence: {''.join(sequence)}")
            print(f"ONNX average decoder time per position: {decoder_time/5:.4f}s")

        # Compare encoder outputs
        print("\nComparing encoder outputs:")
        h_V_diff = np.abs(pt_h_V.cpu().numpy() - onnx_h_V).max()
        h_E_diff = np.abs(pt_h_E.cpu().numpy() - onnx_h_E).max()
        E_idx_diff = np.abs(pt_E_idx.cpu().numpy() - onnx_E_idx).max()

        print(f"Maximum differences:")
        print(f"h_V: {h_V_diff:.6f}")
        print(f"h_E: {h_E_diff:.6f}")
        print(f"E_idx: {E_idx_diff:.6f}")

        # Compare decoder outputs at multiple positions
        print("\nComparing decoder outputs:")
        for pos in range(5):
            print(f"\nPosition {pos}:")
            with torch.no_grad():
                # PyTorch decoder
                pt_logits = decoder(
                    pt_h_V,
                    pt_h_E,
                    pt_E_idx,
                    torch.tensor([pos], device=device),
                    torch.tensor([0.1], device=device)
                )

                # ONNX decoder
                ort_decoder_inputs = {
                    'h_V': onnx_h_V,
                    'h_E': onnx_h_E,
                    'E_idx': onnx_E_idx,
                    'position': np.array([pos], dtype=np.int64),
                    'temperature': np.array([0.1], dtype=np.float32)
                }
                onnx_logits = decoder_session.run(None, ort_decoder_inputs)[0]

                # Compare
                logits_diff = np.abs(pt_logits.cpu().numpy() - onnx_logits).max()
                print(f"Maximum logits difference: {logits_diff:.6f}")

                # Compare top predictions
                pt_probs = torch.softmax(torch.tensor(pt_logits[0, :20]) / 0.1, dim=-1)
                onnx_probs = torch.softmax(torch.tensor(onnx_logits[0, :20]) / 0.1, dim=-1)

                pt_top_probs, pt_top_indices = torch.topk(pt_probs, k=3)
                onnx_top_probs, onnx_top_indices = torch.topk(onnx_probs, k=3)

                print("\nTop 3 predictions comparison:")
                print(f"{'AA':<4} {'PyTorch':>10} {'ONNX':>10} {'Diff':>10}")
                print("-" * 35)
                for i in range(3):
                    pt_aa = aa_dict[pt_top_indices[i].item()]
                    onnx_aa = aa_dict[onnx_top_indices[i].item()]
                    pt_p = pt_top_probs[i].item()
                    onnx_p = onnx_top_probs[i].item()
                    print(f"{pt_aa:<4} {pt_p:>10.4f} {onnx_p:>10.4f} {abs(pt_p-onnx_p):>10.4f}")




if __name__ == "__main__":
    test_ligand_mpnn_sequences()
    export_ligand_mpnn_decoder()
    # test_and_export_ligand_mpnn()
    # test_ligand_feature_extractor()
    # load_ligand_mpnn()
    # test_full_pipeline()
    test_ligand_mpnn_full()
