#!/usr/bin/env python3
"""Clear cache utility."""
import shutil
import argparse
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_cache_size(cache_dir: Path) -> float:
    """Get total size of cache directory in MB."""
    total_size = 0
    for item in cache_dir.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size / (1024 * 1024)


def list_caches():
    """List all cache entries."""
    cache_dir = PROJECT_ROOT / "cache"

    if not cache_dir.exists():
        print("No cache directory found.")
        return

    print("=" * 70)
    print("Cache Entries")
    print("=" * 70)

    total_size = 0
    cache_found = False

    for cache_subdir in cache_dir.iterdir():
        if not cache_subdir.is_dir():
            continue

        cache_name = cache_subdir.name
        size_mb = get_cache_size(cache_subdir)

        if size_mb > 0:
            cache_found = True
            print(f"\n{cache_name}:")
            print(f"  Size: {size_mb:.2f} MB")
            total_size += size_mb

    if cache_found:
        print(f"\nTotal cache size: {total_size:.2f} MB")
    else:
        print("\nNo cache entries found.")

    print("=" * 70)


def clear_cache(cache_name=None):
    """Clear cache."""
    cache_dir = PROJECT_ROOT / "cache"

    if not cache_dir.exists():
        print("No cache directory found.")
        return

    if cache_name:
        target_dir = cache_dir / cache_name
        if target_dir.exists():
            print(f"Clearing cache: {cache_name}")
            shutil.rmtree(target_dir)
            print(f"✓ Cache '{cache_name}' cleared successfully.")
        else:
            print(f"Cache '{cache_name}' not found.")
    else:
        print("Clearing all caches...")
        shutil.rmtree(cache_dir)
        print("✓ All caches cleared successfully.")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Cache management utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all cache entries
  python pipelines/utils/clear_cache.py --list

  # Clear all caches
  python pipelines/utils/clear_cache.py --clear

  # Clear specific cache
  python pipelines/utils/clear_cache.py --clear dataset_splits
        """
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all cache entries'
    )
    parser.add_argument(
        '--clear', '-c',
        nargs='?',
        const='__all__',
        metavar='CACHE_NAME',
        help='Clear cache (specify name or leave empty to clear all)'
    )

    args = parser.parse_args()

    if args.list:
        list_caches()
    elif args.clear:
        if args.clear == '__all__':
            clear_cache()
        else:
            clear_cache(args.clear)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
