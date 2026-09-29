"""Full q65 pipeline: discover active fires, fetch data, forecast, arrival time, exports.

    python pipeline.py                  # every qualifying active WFIGS fire
    python pipeline.py --fire "Dome"    # one fire
"""
import argparse

from fetchData import cacheDir, discover, fetchFire
from runModel import loadModels, outputDir, runFire


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fire", help="WFIGS incident name (default: every active fire)")
    parser.add_argument("--cache", default=cacheDir)
    parser.add_argument("--out", default=outputDir)
    args = parser.parse_args()

    models = loadModels()
    fires = discover(args.fire)
    print(f"{len(fires)} fire(s): {[f['name'] for f in fires]}")
    failed = 0
    for fire in fires:
        print(f"\n{fire['name']}")
        try:
            runFire(fetchFire(fire, args.cache), models, args.out)
        except Exception as error:  # one fire must not stop the batch
            failed += 1
            print(f"  failed: {type(error).__name__}: {error}")
    print(f"\n{len(fires) - failed}/{len(fires)} fire(s) done")


if __name__ == "__main__":
    main()
