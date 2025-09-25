#!/usr/bin/env python3
"""
Test script to verify the line masking works with the actual configuration.
"""
import os
import sys
import torch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_integration():
    """Test the line replacement masking with the actual config."""
    print("Testing line replacement masking with span_and_lines config...")
    
    # Load the actual config
    config_path = os.path.join(os.path.dirname(__file__), 'data', 'char_diffusion', 'config', 'span_and_lines.py')
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("span_and_lines_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract configuration
    config = {
        'use_all_stages_for_training': config_module.use_all_stages_for_training,
        'unmasking_stages': config_module.unmasking_stages,
        'validation_stages': config_module.validation_stages,
        'enable_line_aligned_sequences': True,
        'mask_probability': 0.15
    }
    
    print(f"Loaded config with {len(config['unmasking_stages'])} unmasking stages")
    print(f"Unmasking stages: {config['unmasking_stages']}")
    
    # Find the line stage
    line_stage = None
    for stage in config['unmasking_stages']:
        if stage['type'] == 'line':
            line_stage = stage
            break
    
    if line_stage is None:
        print("❌ No line stage found in configuration!")
        return False
    
    print(f"Found line stage: {line_stage}")
    
    # Initialize provider with the actual config
    from data.char_diffusion.prepare_streaming import CharDiffusionProvider
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'char_diffusion')
    
    try:
        provider = CharDiffusionProvider(
            data_dir=data_dir,
            batch_size=8,
            block_size=256,
            batches_per_file=5,
            max_backlog_files=1,
            sleep_seconds=1.0,
            seed=1337,
            verbose=True,
            config=config
        )
        
        print(f"✅ Provider initialized successfully with full config!")
        print(f"Training stages: {len(provider.train_stage_distribution)}")
        
        # Test that the line stage is properly distributed
        for i, stage_info in enumerate(provider.train_stage_distribution):
            stage_config = stage_info['config']
            count = stage_info['count']
            stage_type = stage_config.get('type', 'unknown')
            print(f"  Stage {i}: {stage_type} -> {count} samples per file")
            
            if stage_type == 'line':
                min_ratio = stage_config.get('min_ratio', 'unknown')
                max_ratio = stage_config.get('max_ratio', 'unknown')
                print(f"    Line replacement ratio: {min_ratio} - {max_ratio}")
        
        # Test stage-based file generation
        print("\n🧪 Testing stage-based file generation with line masking...")
        provider._produce_stage_based_file('train', 0)
        print("✅ Stage-based file generation completed!")
        
        # Load and inspect the generated file
        train_files = os.listdir(provider.train_dir)
        if train_files:
            latest_file = max(train_files, key=lambda f: os.path.getctime(os.path.join(provider.train_dir, f)))
            file_path = os.path.join(provider.train_dir, latest_file)
            
            print(f"\n📁 Inspecting generated file: {latest_file}")
            data = torch.load(file_path)
            
            tensors = data['tensors']
            metadata = data['metadata']
            
            print(f"  Batch size: {tensors['x'].shape[0]}")
            print(f"  Sequence length: {tensors['x'].shape[1]}")
            print(f"  Stage info available: {'stage_info' in metadata}")
            
            if 'stage_info' in metadata:
                stage_types = [stage.get('type', 'unknown') for stage in metadata['stage_info']]
                stage_counts = {}
                for stage_type in stage_types:
                    stage_counts[stage_type] = stage_counts.get(stage_type, 0) + 1
                
                print(f"  Stage distribution in file: {stage_counts}")
                
                # Check if line masking was applied
                if 'line' in stage_counts:
                    print(f"  ✅ Line masking was applied to {stage_counts['line']} samples")
                else:
                    print(f"  ⚠️  No line masking found in this file")
        
        print("\n✅ Configuration integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_config_integration()
    if success:
        print("\n🎉 All integration tests passed!")
    else:
        print("\n💥 Integration tests failed!")
        sys.exit(1)
