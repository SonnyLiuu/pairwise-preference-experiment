from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk

from engines.cars_eubo import CarsEUBOEngine
from engines.cars_bald import CarsBALDEngine
from engines.gambles_eubo import GamblesEUBOEngine
from engines.gambles_bald import GamblesBALDEngine

CAR_BRANDS = ["Benz", "Toyota", "Ford"]
CAR_COLORS = ["Black", "White", "Red"]
CAR_FUELS  = ["Hybrid", "Gas", "Electric"]


HEADER = [
    "subject_id",
    "block_index",
    "block_label",
    "trial_in_block",
    "timestamp_iso",
    "rt_seconds",
    "choice",
    "model_p_left",
    "model_p_right",
    "left_brand_idx",
    "left_color_idx",
    "left_fuel_idx",
    "right_brand_idx",
    "right_color_idx",
    "right_fuel_idx",
    "left_brand",
    "left_color",
    "left_fuel",
    "right_brand",
    "right_color",
    "right_fuel",
    "left_p_win",
    "left_win_amt",
    "left_loss_amt",
    "right_p_win",
    "right_win_amt",
    "right_loss_amt",
]


def ensure_data_file(subject_id: str) -> Path:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    path = data_dir / f"subject_{subject_id}.csv"
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
    return path


def log_trial(
    csv_path: Path,
    subject_id: str,
    block_index: int,
    trial,
    rt_seconds: float,
    choice: str,
) -> None:
    timestamp_iso = datetime.now().isoformat()
    block_label = getattr(trial, "block_label", "")

    left_brand_idx = right_brand_idx = ""
    left_color_idx = right_color_idx = ""
    left_fuel_idx  = right_fuel_idx  = ""
    left_brand = left_color = left_fuel = ""
    right_brand = right_color = right_fuel = ""

    left_p_win = left_win_amt = left_loss_amt = ""
    right_p_win = right_win_amt = right_loss_amt = ""

    left_features = getattr(trial, "left_features", None)
    right_features = getattr(trial, "right_features", None)

    if block_label.startswith("cars_") and left_features is not None and right_features is not None:
        lf = left_features
        rf = right_features

        lb = int(round(float(lf[0].item())))
        lc = int(round(float(lf[1].item())))
        lfuel = int(round(float(lf[2].item())))

        rb = int(round(float(rf[0].item())))
        rc = int(round(float(rf[1].item())))
        rfuel = int(round(float(rf[2].item())))

        left_brand_idx  = lb
        left_color_idx  = lc
        left_fuel_idx   = lfuel
        right_brand_idx = rb
        right_color_idx = rc
        right_fuel_idx  = rfuel

        left_brand  = CAR_BRANDS[lb]
        left_color  = CAR_COLORS[lc]
        left_fuel   = CAR_FUELS[lfuel]
        right_brand = CAR_BRANDS[rb]
        right_color = CAR_COLORS[rc]
        right_fuel  = CAR_FUELS[rfuel]

    elif block_label.startswith("gambles_") and left_features is not None and right_features is not None:
        lf = left_features
        rf = right_features

        left_p_win    = float(lf[0].item())
        left_win_amt  = float(lf[1].item())
        left_loss_amt = float(lf[2].item())

        right_p_win    = float(rf[0].item())
        right_win_amt  = float(rf[1].item())
        right_loss_amt = float(rf[2].item())

    model_p_left = getattr(trial, "model_p_left", None)
    model_p_right = getattr(trial, "model_p_right", None)

    row = [
        subject_id,
        block_index,
        block_label,
        trial.trial_in_block,
        timestamp_iso,
        rt_seconds,
        choice,
        model_p_left if model_p_left is not None else "",
        model_p_right if model_p_right is not None else "",
        left_brand_idx,
        left_color_idx,
        left_fuel_idx,
        right_brand_idx,
        right_color_idx,
        right_fuel_idx,
        left_brand,
        left_color,
        left_fuel,
        right_brand,
        right_color,
        right_fuel,
        left_p_win,
        left_win_amt,
        left_loss_amt,
        right_p_win,
        right_win_amt,
        right_loss_amt,
    ]

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


class ExperimentApp:
    """Full-screen Tkinter app for pairwise preference experiments."""

    def __init__(
        self,
        root: tk.Tk,
        subject_id: str,
        csv_path: Path,
        block_order: list[str],
    ) -> None:
        self.root = root
        self.subject_id = subject_id
        self.csv_path = csv_path
        self.block_order = block_order
        self.block_index = 0

        self.engine = None
        self.current_trial = None
        self.trial_start_time = None
        self.choice_locked = False

        root.title("Preference Experiment")
        root.attributes("-fullscreen", True)
        root.configure(bg="black")
        root.focus_force()
        root.bind("<Key>", self.on_key)

        self.question_label = tk.Label(
            root,
            text="Which option do you prefer?",
            font=("Arial", 28),
            bg="black",
            fg="white",
            wraplength=900,
            justify="center",
        )
        self.question_label.place(relx=0.5, rely=0.25, anchor="center")

        self.left_button = tk.Button(
            root,
            text="",
            font=("Arial", 24),
            bg="#222222",
            fg="white",
            disabledforeground="white",
            activebackground="#222222",
            activeforeground="white",
            wraplength=500,
            justify="center",
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=40,
            pady=25,
            width=18,
            command=lambda: self.handle_choice("left"),
        )
        self.left_button.place(relx=0.25, rely=0.7, anchor="center")

        self.right_button = tk.Button(
            root,
            text="",
            font=("Arial", 24),
            bg="#222222",
            fg="white",
            disabledforeground="white",
            activebackground="#222222",
            activeforeground="white",
            wraplength=500,
            justify="center",
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=40,
            pady=25,
            width=18,
            command=lambda: self.handle_choice("right"),
        )
        self.right_button.place(relx=0.75, rely=0.7, anchor="center")

        self.info_label = tk.Label(
            root,
            text="Click LEFT or RIGHT. Press Esc to quit.",
            font=("Arial", 14),
            bg="black",
            fg="gray",
        )
        self.info_label.place(relx=0.5, rely=0.93, anchor="center")

        self.start_next_block()

    def start_next_block(self):
        if self.block_index >= len(self.block_order):
            self.end_experiment()
            return

        block_label = self.block_order[self.block_index]
        print(f"Starting block {self.block_index + 1}: {block_label}")

        if block_label == "cars_eubo":
            self.engine = CarsEUBOEngine(block_label=block_label)
        elif block_label == "cars_bald":
            self.engine = CarsBALDEngine(block_label=block_label)
        elif block_label == "gambles_eubo":
            self.engine = GamblesEUBOEngine(block_label=block_label)
        elif block_label == "gambles_bald":
            self.engine = GamblesBALDEngine(block_label=block_label)
        else:
            raise ValueError(f"Unknown block label: {block_label!r}")

        if block_label.startswith("cars_"):
            q = "Which car would you rather have?"
        elif block_label.startswith("gambles_"):
            q = "Which option would you rather play?"
        else:
            q = "Which option do you prefer?"

        self.question_label.config(text=q)
        self.current_trial = self.engine.start()
        self.show_trial(self.current_trial)

    def show_trial(self, trial):
        self.choice_locked = False

        self.left_button.config(
            text=trial.left_display,
            bg="#222222",
            state="normal",
        )
        self.right_button.config(
            text=trial.right_display,
            bg="#222222",
            state="normal",
        )

        self.info_label.config(
            text=f"Block {self.block_index+1}/{len(self.block_order)} — "
                 f"Trial {trial.trial_in_block}"
        )

        self.root.update_idletasks()
        self.root.focus_force()
        self.trial_start_time = time.perf_counter()

    def on_key(self, event):
        if event.keysym.lower() == "escape":
            self.root.destroy()

    def handle_choice(self, side: str):
        if self.choice_locked:
            return
        if self.current_trial is None or self.trial_start_time is None:
            return
        if side not in ("left", "right"):
            return

        self.choice_locked = True

        self.left_button.config(state="disabled")
        self.right_button.config(state="disabled")

        if side == "left":
            self.left_button.config(bg="#1E7F4E")
        else:
            self.right_button.config(bg="#1E7F4E")

        self.root.update_idletasks()
        self.root.update()

        rt = time.perf_counter() - self.trial_start_time

        self.root.after(150, lambda s=side, r=rt: self._finalize_choice(s, r))

    def _finalize_choice(self, side: str, rt: float):
        if self.current_trial is None or self.engine is None:
            return

        log_trial(
            csv_path=self.csv_path,
            subject_id=self.subject_id,
            block_index=self.block_index + 1,
            trial=self.current_trial,
            rt_seconds=rt,
            choice=side,
        )

        self.engine.update(side)
        next_trial = self.engine.next_trial()

        if next_trial is None:
            self.end_block()
        else:
            self.current_trial = next_trial
            self.show_trial(next_trial)

    def end_block(self):
        print(f"Finished block {self.block_index + 1}")
        self.block_index += 1
        self.engine = None
        self.current_trial = None

        self.left_button.config(text="", state="disabled", bg="#222222")
        self.right_button.config(text="", state="disabled", bg="#222222")
        self.question_label.config(text="Block complete.")
        self.info_label.config(text="Next block starting...")
        self.choice_locked = True

        self.root.after(1500, self.start_next_block)

    def end_experiment(self):
        print("Experiment finished.")
        self.left_button.config(state="disabled")
        self.right_button.config(state="disabled")
        self.question_label.config(text="Thank you for participating.")
        self.info_label.config(text="")
        self.choice_locked = True
        self.root.after(2000, self.root.destroy)


def main() -> None:
    subject_id = input("Enter subject ID: ").strip()
    if not subject_id:
        print("No subject ID provided; exiting.")
        return

    csv_path = ensure_data_file(subject_id)

    block_order = ["cars_eubo", "cars_bald", "gambles_eubo", "gambles_bald"]

    root = tk.Tk()
    app = ExperimentApp(
        root=root,
        subject_id=subject_id,
        csv_path=csv_path,
        block_order=block_order,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
