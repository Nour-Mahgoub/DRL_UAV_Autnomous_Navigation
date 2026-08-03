import os
import datetime


class TrainingLogger:
    """
    Writes a comprehensive, comparison-ready .log file for a single training run.

    Format: a commented header block (config/hyperparameters, one per line,
    prefixed with '#') followed by a CSV table of per-episode metrics. This
    keeps the file human-readable while also being directly loadable later via:

        import pandas as pd
        df = pd.read_csv("results/logs/<run_name>.log", comment="#")

    which makes building comparison plots across multiple runs straightforward.

    Usage:
        logger = TrainingLogger()          # prompts for a run name
        logger.log_config({...})           # log hyperparameters/settings once
        logger.log_episode(episode=1, reward=-4.11, avg10=-4.11,
                            success_rate10=0.0, steps=150, epsilon=0.996,
                            elapsed_min=1.7)
        logger.log_event("Training complete.")
        logger.close()
    """

    def __init__(self, run_name=None, results_dir=None):
        if results_dir is None:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
        os.makedirs(results_dir, exist_ok=True)

        if run_name is None:
            run_name = input("Enter a name for this run (e.g. 'd3qn_v3_time_penalty'): ").strip()
            if not run_name:
                run_name = "run"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        filename = f"{run_name}_{timestamp}.log"
        self.log_path = os.path.join(results_dir, filename)

        self._file = open(self.log_path, "w", encoding="utf-8")
        self._file.write(f"# run_name: {run_name}\n")
        self._file.write(f"# timestamp: {timestamp}\n")
        self._header_written = False

        print(f"Logging this run to: {self.log_path}")

    def log_config(self, config: dict):
        """Log hyperparameters/settings as commented key: value lines. Call once, before training starts."""
        self._file.write("# ---- config ----\n")
        for key, value in config.items():
            self._file.write(f"# {key}: {value}\n")
        self._file.write("# ----------------\n")
        self._file.flush()

    def log_event(self, message: str):
        """Log a free-text note (warnings, milestones, manual observations) as a commented line."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._file.write(f"# [{timestamp}] {message}\n")
        self._file.flush()

    def log_episode(self, episode, reward, avg10, success_rate10, steps, epsilon, elapsed_min, **extra):
        """
        Log one episode's metrics as a CSV row. Extra keyword args are
        appended as additional columns (e.g. loss=..., termination_reason=...).
        The header row is written automatically on the first call, including
        any extra columns present at that point.
        """
        base_fields = {
            "episode": episode,
            "reward": reward,
            "avg10": avg10,
            "success_rate10": success_rate10,
            "steps": steps,
            "epsilon": epsilon,
            "elapsed_min": elapsed_min,
        }
        base_fields.update(extra)

        if not self._header_written:
            self._file.write(",".join(base_fields.keys()) + "\n")
            self._header_written = True
            self._columns = list(base_fields.keys())

        # keep column order consistent even if some episodes omit optional extras
        row = [str(base_fields.get(col, "")) for col in self._columns]
        self._file.write(",".join(row) + "\n")
        self._file.flush()  # flush every episode — don't lose data if training crashes

    def close(self):
        self._file.close()
        print(f"Log saved: {self.log_path}")