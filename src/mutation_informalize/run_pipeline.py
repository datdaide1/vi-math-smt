"""
Main Pipeline: Run all phases sequentially.
Each phase is fully resumeable — safe to Ctrl+C and re-run.

NEW ARCHITECTURE (4 Phases):
Phase 1: Validate + Classify + Fix SMT
Phase 2: Mutate SMT (Pure Algorithm with Smart Retry)
Phase 3: Informalize (Agent)
Phase 4: Merge

Usage:
    cd src/mutation_informalize
    
    # Run everything:
    python run_pipeline.py
    
    # Run specific phase:
    python run_pipeline.py --phase 1   # Validate + Fix SMT
    python run_pipeline.py --phase 2   # Mutation with retry
    python run_pipeline.py --phase 3   # Informalize
    python run_pipeline.py --phase 4   # Merge
"""
import asyncio
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_async(coro):

    try:

        loop = asyncio.get_running_loop()

        return loop.create_task(coro)

    except RuntimeError:

        return asyncio.run(coro)


def run_phase(phase_num: int):

    try:

        if phase_num == 1:

            from phase1_validate import run_phase1

            run_async(run_phase1())

        elif phase_num == 2:

            from phase2_mutate import run_phase2

            run_phase2()

        elif phase_num == 3:

            from phase3_informalize import run_phase3

            run_async(run_phase3())

        elif phase_num == 4:

            from phase4_merge import run_phase4

            run_phase4()

        else:

            print(f"Unknown phase: {phase_num}")

    except Exception as e:

        print(f"\n[ERROR] Phase {phase_num} failed")

        print(str(e))

        raise


def main():

    parser = argparse.ArgumentParser(
        description="Mutation-Informalize Pipeline (NEW)"
    )

    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        help="Run specific phase (1-4). Default: run all."
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  MUTATION-INFORMALIZE PIPELINE (NEW ARCHITECTURE)")
    print("  Phase 1: Validate + Classify + Fix SMT")
    print("  Phase 2: Mutate SMT (Pure Algorithm with Smart Retry)")
    print("  Phase 3: Informalize (Agent)")
    print("  Phase 4: Merge")
    print("  Each phase is resumeable. Safe to Ctrl+C and re-run.")
    print("=" * 70)

    start = time.time()

    try:

        if args.phase > 0:

            phase_start = time.time()

            run_phase(args.phase)

            phase_elapsed = time.time() - phase_start

            print(
                f"\n  Phase {args.phase} "
                f"completed in {phase_elapsed:.1f}s"
            )

        else:

            for phase in [1, 2, 3, 4]:

                print(f"\n{'#' * 70}")
                print(f"  PHASE {phase}")
                print(f"{'#' * 70}\n")

                phase_start = time.time()

                run_phase(phase)

                phase_elapsed = time.time() - phase_start

                print(
                    f"\n  Phase {phase} "
                    f"completed in {phase_elapsed:.1f}s"
                )

    except KeyboardInterrupt:

        print("\n\nPipeline interrupted safely.")

        sys.exit(0)

    elapsed = time.time() - start

    hours = int(elapsed // 3600)

    minutes = int((elapsed % 3600) // 60)

    seconds = int(elapsed % 60)

    print(f"\n  ✓ All phases complete!")

    print(
        f"  Total time: "
        f"{hours}h {minutes}m {seconds}s"
    )


if __name__ == "__main__":
    main()