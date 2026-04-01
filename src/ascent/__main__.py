import ascent.cli

if __name__ == "__main__":
    # 1. CRITICAL: Set sharing strategy BEFORE any other imports or logic
    # 'file_system' is much more stable on HPC clusters than 'file_descriptor'

    # 2. Run the actual application

    ascent.cli.main()
