import argparse
from environments import SimulatedSpe_edEnv
from policies import RandomPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='spe_ed')
    parser.add_argument('--render', action='store_true', help='Render simulation.')
    args = parser.parse_args()

    env = SimulatedSpe_edEnv(40, 40, [RandomPolicy() for _ in range(5)])
    pol = RandomPolicy()

    obs = env.reset()
    done = False

    while not done:
        if args.render:
            for _ in range(10):
                if not env.render(screen_width=720, screen_height=720):
                    break

        action = pol.act(*obs)
        obs, reward, done, _ = env.step(action)
    print(f"Reward: {reward}")

    if args.render:
        # Show final state
        for _ in range(60 * 5):
            if not env.render(screen_width=720, screen_height=720):
                break
