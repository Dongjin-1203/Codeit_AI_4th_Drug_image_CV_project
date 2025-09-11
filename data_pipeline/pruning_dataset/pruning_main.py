from modules.dataset_splitter import create_small_dataset
from modules.config import get_config, list_all_configs

# =============================================================================
# Quick Creation Functions
# =============================================================================

def create_by_config(config_name):
    """Create dataset using predefined config"""
    config = get_config(config_name)
    create_small_dataset(
        source_path="./data",
        output_path=config['output_path'],
        train_size=config['train_size'],
        test_size=config['test_size'],
        sampling_strategy=config['sampling_strategy']
    )
    print(f"Dataset ready at {config['output_path']}")

def create_custom(train_size, test_size, output_name="custom", strategy="balanced"):
    """Create custom-sized dataset"""
    output_path = f"./data/{output_name}_data"
    create_small_dataset(
        source_path="./data",
        output_path=output_path,
        train_size=train_size,
        test_size=test_size,
        sampling_strategy=strategy
    )
    print(f"Custom dataset ready at {output_path}")

# =============================================================================
# Predefined Quick Functions
# =============================================================================

def quick_prototype():
    """Prototype dataset (50/25)"""
    create_by_config("prototype")

def quick_dev():
    """Development dataset (150/75)"""
    create_by_config("development")

def quick_experiment():
    """Experiment dataset (300/150)"""
    create_by_config("experiment")

def quick_demo():
    """Demo dataset (100/50)"""
    create_by_config("demo")

# =============================================================================
# Batch Creation
# =============================================================================

def create_standard_set():
    """Create standard datasets for team"""
    configs = ["prototype", "development", "experiment", "demo"]
    
    print(f"Creating {len(configs)} standard datasets...")
    for config_name in configs:
        print(f"\nCreating {config_name}...")
        create_by_config(config_name)
    
    print("\nStandard dataset set complete!")

def create_experiment_set():
    """Create datasets for model experiments"""
    configs = ["prototype", "experiment", "validation"]
    
    print("Creating experiment dataset set...")
    for config_name in configs:
        print(f"\nCreating {config_name}...")
        create_by_config(config_name)
    
    print("\nExperiment dataset set complete!")

# =============================================================================
# Interactive Menu
# =============================================================================

def show_menu():
    """Show available options"""
    print("\nDataset Creation Menu")
    print("=" * 30)
    
    options = {
        '1': ('Prototype (50/25)', quick_prototype),
        '2': ('Development (150/75)', quick_dev),
        '3': ('Experiment (300/150)', quick_experiment),
        '4': ('Demo (100/50)', quick_demo),
        '5': ('Standard Set', create_standard_set),
        '6': ('Experiment Set', create_experiment_set),
        '7': ('List Configs', list_all_configs),
        '8': ('Custom Size', _custom_menu)
    }
    
    for key, (desc, _) in options.items():
        print(f"  {key}. {desc}")
    
    choice = input("\nSelect (1-8): ").strip()
    
    if choice in options:
        _, func = options[choice]
        if func:
            func()
    else:
        print("Invalid choice")

def _custom_menu():
    """Custom dataset creation menu"""
    try:
        train_size = int(input("Train size: "))
        test_size = int(input("Test size: "))
        output_name = input("Output name (default: custom): ").strip() or "custom"
        
        strategies = ["balanced", "quality", "random"]
        print(f"Strategies: {', '.join(strategies)}")
        strategy = input("Strategy (default: balanced): ").strip() or "balanced"
        
        if strategy in strategies:
            create_custom(train_size, test_size, output_name, strategy)
        else:
            print("Invalid strategy")
    except ValueError:
        print("Invalid input")

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        # Direct function calls
        commands = {
            'prototype': quick_prototype,
            'dev': quick_dev,
            'experiment': quick_experiment,
            'demo': quick_demo,
            'standard': create_standard_set,
            'exp_set': create_experiment_set,
            'list': list_all_configs,
            'menu': show_menu
        }
        
        if command in commands:
            commands[command]()
        elif command == 'custom':
            # Custom creation: python quick_script.py custom 200 100 my_dataset balanced
            if len(sys.argv) >= 4:
                try:
                    train_size = int(sys.argv[2])
                    test_size = int(sys.argv[3])
                    output_name = sys.argv[4] if len(sys.argv) > 4 else "custom"
                    strategy = sys.argv[5] if len(sys.argv) > 5 else "balanced"
                    create_custom(train_size, test_size, output_name, strategy)
                except ValueError:
                    print("Usage: python quick_script.py custom <train_size> <test_size> [output_name] [strategy]")
            else:
                print("Usage: python quick_script.py custom <train_size> <test_size> [output_name] [strategy]")
        else:
            print(f"Unknown command: {command}")
            print(f"Available: {list(commands.keys())}")
    else:
        # No arguments - show menu
        show_menu()

if __name__ == "__main__":
    main()