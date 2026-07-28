from pathlib import Path
from game.base import Game

def main():
    """Main entry point demonstrating StateManager usage."""
    
    # 1. Create the Game instance
    game = Game(
        base_shader_path=Path("game/shaders/background.frag"),
        title=b"Yahtzee RL Simulation",
        target_fps=60
    )
    
    # 2. Initialize the game (window, shaders, textures)
    game.initialize()
    
    # 3. Setup states (registers states with the StateManager)
    game.setup_states()
    
    # 4. Run the game loop (StateManager handles state updates/rendering)
    game.run_loop()
    
    # 5. Cleanup
    game.shutdown()

if __name__ == "__main__":
    main()