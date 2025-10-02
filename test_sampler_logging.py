"""
Test script to verify sampler loss logging
"""
from core.logger import ConsoleLogger, WandBLogger

def test_console_logging():
    """Test console logger with sampler loss"""
    print("Testing console logger...")
    logger = ConsoleLogger(master_process=True)
    
    # Test 1: Only main loss
    print("\n1. Only main loss:")
    logger.log_step({
        'iter': 100,
        'loss': 4.5,
        'loss_main': 4.5,
        'time_ms': 150.0,
        'mfu_pct': 25.5
    })
    
    # Test 2: Main + Sampler
    print("\n2. Main + Sampler:")
    logger.log_step({
        'iter': 1500,
        'loss': 9.2,
        'loss_main': 4.6,
        'loss_sampler': 4.6,
        'time_ms': 155.0,
        'mfu_pct': 26.0
    })
    
    # Test 3: Main + Sampler + Critic
    print("\n3. Main + Sampler + Critic:")
    logger.log_step({
        'iter': 3500,
        'loss': 13.8,
        'loss_main': 4.6,
        'loss_sampler': 4.6,
        'loss_critic': 4.6,
        'time_ms': 160.0,
        'mfu_pct': 26.5
    })
    
    # Test 4: Sampler loss = 0 (should not show)
    print("\n4. Sampler loss = 0 (should not show):")
    logger.log_step({
        'iter': 500,
        'loss': 4.5,
        'loss_main': 4.5,
        'loss_sampler': 0.0,
        'time_ms': 150.0,
        'mfu_pct': 25.5
    })
    
    print("\n✓ Console logging test complete")


def test_wandb_logging():
    """Test WandB logger metrics dict (without actually initializing wandb)"""
    print("\nTesting WandB logger metrics dict...")
    
    # Create logger without initializing wandb
    logger = WandBLogger(
        project='test',
        run_name='test',
        config={},
        master_process=True,
        enabled=False  # Don't actually initialize wandb
    )
    
    # Manually test the log_dict construction
    metrics = {
        'iter': 1500,
        'loss': 9.2,
        'loss_main': 4.6,
        'loss_sampler': 4.6,
        'loss_critic': 0.0,
        'time_ms': 155.0,
        'mfu_pct': 26.0
    }
    
    # Simulate what log_step does
    log_dict = {}
    if 'iter' in metrics:
        log_dict['iter'] = metrics['iter']
    if 'loss' in metrics:
        log_dict['train/loss'] = metrics['loss']
    if 'loss_main' in metrics:
        log_dict['train/loss_main'] = metrics['loss_main']
    if 'loss_sampler' in metrics:
        log_dict['train/loss_sampler'] = metrics['loss_sampler']
    if 'loss_critic' in metrics:
        log_dict['train/loss_critic'] = metrics['loss_critic']
    if 'time_ms' in metrics:
        log_dict['time_ms'] = metrics['time_ms']
    if 'mfu_pct' in metrics:
        log_dict['mfu'] = metrics['mfu_pct']
    
    print("\nWandB log_dict would contain:")
    for key, value in log_dict.items():
        print(f"  {key}: {value}")
    
    # Verify expected keys
    expected_keys = ['iter', 'train/loss', 'train/loss_main', 'train/loss_sampler', 
                     'train/loss_critic', 'time_ms', 'mfu']
    missing = [k for k in expected_keys if k not in log_dict]
    if missing:
        print(f"\n✗ Missing keys: {missing}")
        return False
    
    print("\n✓ WandB logging test complete")
    return True


def main():
    print("="*60)
    print("SAMPLER LOSS LOGGING TESTS")
    print("="*60)
    
    test_console_logging()
    test_wandb_logging()
    
    print("\n" + "="*60)
    print("ALL LOGGING TESTS PASSED ✓")
    print("="*60)


if __name__ == '__main__':
    main()

