"""
Graphical interface for single-sample inference with the trained ensemble + meta-model
pipeline described in the accompanying paper.

The user provides the chemical/product composition ('productcomp') and a small set of
sampling-related fields; the product type and product specification fields are filled in
automatically from the majority (producttype, productspec) combination observed for the given
productcomp in the training data, and are not directly editable. Submitting the form runs the
full base-model + meta-model pipeline and displays the top-3 most likely target classes
('paramf'), decoded back to their original string labels.

A second entry point, the 'Test real data' button, draws one random row from a held-out sample
of real test rows, fills the whole form with it, runs the same pipeline, and displays the
predicted classes alongside the actual ('paramf') label, for a quick qualitative check of the
model's behaviour on genuine data.

Expected directory layout (relative to this file):
    models/     model1.pkl ... modelN.pkl, meta_model.pkl (or .txt), meta_model_metadata.pkl
    encoders/   features_encoder.pkl, target_encoder.pkl
    columns/    one '<column>.txt' per user-editable column, plus completion.csv
    data/       test_meta_string.xlsx (random sample of real, decoded test rows; optional)
    utils/      io_utils.py, encoding_utils.py, prediction_utils.py, autocomplete_utils.py,
                widgets.py
"""

import os
import tkinter as tk
from tkinter import messagebox, ttk

from utils.autocomplete_utils import get_auto_completed_fields
from utils.encoding_utils import UnseenLabelError
from utils.io_utils import load_all_resources
from utils.prediction_utils import predict_top_k
from utils.widgets import AutocompleteCombobox

# =========================
# CONFIGURATION
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ENCODERS_DIR = os.path.join(BASE_DIR, 'encoders')
COLUMNS_DIR = os.path.join(BASE_DIR, 'columns')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test_meta_string.xlsx')

FEATURE_COLUMNS = [
    'productcomp', 'producttype', 'samppoint', 'labcountry',
    'origcountry', 'sampcountry', 'contamination', 'productspec'
]
TARGET_COLUMN = 'paramf'

# Fields filled in automatically from the completion table; the user cannot edit them directly.
AUTO_FILLED_COLUMNS = ['producttype', 'productspec']
# Fields the user can type into, with autocomplete suggestions drawn from the training data.
EDITABLE_COLUMNS = [col for col in FEATURE_COLUMNS if col not in AUTO_FILLED_COLUMNS]

TOP_K = 3

# Widget sizing (widened slightly with respect to the first version of the GUI).
FIELD_WIDTH = 55
OUTPUT_WIDTH = 65
OUTPUT_HEIGHT = TOP_K + 2


class PredictionApp(tk.Tk):
    """
    Main application window: one input row per feature column, two action buttons ('Predict'
    and 'Test real data'), and an output area showing the top-k predicted target classes.
    """

    def __init__(self, resources):
        """
        INPUT: resources (dict): bundle of fitted models/encoders/lookup tables as returned by
                   utils.io_utils.load_all_resources.
        OUTPUT: None. Builds and lays out all widgets of the main window.
        """
        super().__init__()
        self.resources = resources
        self.title("Chemical Hazard Ranking - Single Sample Prediction")
        self.resizable(False, False)

        self.field_widgets = {}
        self._build_form()
        self._build_action_buttons()
        self._build_output_area()

    # ------------------------------------------------------------------
    # WIDGET CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_form(self):
        """
        INPUT: None.
        OUTPUT: None. Creates one label + combobox pair per feature column and stores the
            combobox references in self.field_widgets for later access.
        """
        form_frame = ttk.Frame(self, padding=15)
        form_frame.grid(row=0, column=0, sticky='nsew')

        for row_index, column_name in enumerate(FEATURE_COLUMNS):
            label = ttk.Label(form_frame, text=column_name)
            label.grid(row=row_index, column=0, sticky='w', pady=3)

            combobox = AutocompleteCombobox(form_frame, width=FIELD_WIDTH)
            combobox.grid(row=row_index, column=1, pady=3, padx=8)

            if column_name in AUTO_FILLED_COLUMNS:
                combobox.configure(state='disabled')
            else:
                values = self.resources['autocomplete_values'].get(column_name, [])
                combobox.set_completion_list(values)

            self.field_widgets[column_name] = combobox

        # Any change to 'productcomp' triggers the auto-completion of the dependent fields.
        self.field_widgets['productcomp'].bind('<KeyRelease>', self._on_productcomp_change)
        self.field_widgets['productcomp'].bind('<<ComboboxSelected>>', self._on_productcomp_change)

    def _build_action_buttons(self):
        """
        INPUT: None.
        OUTPUT: None. Adds the two action buttons: 'Predict' (runs the pipeline on the values
            currently in the form) and 'Test real data' (draws and predicts a random row from
            the held-out test sample, showing the real label alongside the prediction).
        """
        buttons_frame = ttk.Frame(self)
        buttons_frame.grid(row=1, column=0, pady=(0, 10))

        predict_button = ttk.Button(buttons_frame, text="Predict", command=self._on_submit)
        predict_button.grid(row=0, column=0, padx=8)

        test_button = ttk.Button(
            buttons_frame, text="Test real data", command=self._on_test_real_data
        )
        test_button.grid(row=0, column=1, padx=8)

    def _build_output_area(self):
        """
        INPUT: None.
        OUTPUT: None. Adds the read-only text area used to display the top-k predictions (and,
            when available, the real label for comparison).
        """
        output_frame = ttk.LabelFrame(
            self, text=f"Top-{TOP_K} predicted '{TARGET_COLUMN}'", padding=10
        )
        output_frame.grid(row=2, column=0, sticky='nsew', padx=15, pady=(0, 15))

        self.output_text = tk.Text(
            output_frame, height=OUTPUT_HEIGHT, width=OUTPUT_WIDTH, state='disabled'
        )
        self.output_text.pack()

    # ------------------------------------------------------------------
    # FORM HELPERS
    # ------------------------------------------------------------------

    def _fill_fields(self, input_row):
        """
        INPUT: input_row (dict[str, str]): feature values to display in the form, keyed by
                   column name.
        OUTPUT: None. Writes each value into its corresponding widget, going through
            _set_readonly_field for the auto-filled columns so they remain non-editable
            afterwards.
        """
        for column_name in FEATURE_COLUMNS:
            value = input_row[column_name]
            if column_name in AUTO_FILLED_COLUMNS:
                self._set_readonly_field(column_name, value)
            else:
                self.field_widgets[column_name].set(value)

    def _set_readonly_field(self, column_name, value):
        """
        INPUT: column_name (str): name of a read-only (auto-filled) field.
               value (str): text to display in the field.
        OUTPUT: None. Temporarily re-enables the widget to update its content, then disables it
            again so the user cannot edit it directly.
        """
        widget = self.field_widgets[column_name]
        widget.configure(state='normal')
        widget.set(value)
        widget.configure(state='disabled')

    # ------------------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------------------

    def _on_productcomp_change(self, event):
        """
        INPUT: event (tkinter.Event): key-release or selection event on the 'productcomp' field.
        OUTPUT: None. Looks up the majority (producttype, productspec) pair for the current
            'productcomp' text and writes it into the corresponding read-only fields; clears
            them if no match is found in the completion table.
        """
        productcomp_value = self.field_widgets['productcomp'].get()
        completion_table = self.resources['completion_table']
        producttype_value, productspec_value = get_auto_completed_fields(
            productcomp_value, completion_table
        )
        self._set_readonly_field('producttype', producttype_value or '')
        self._set_readonly_field('productspec', productspec_value or '')

    def _on_submit(self):
        """
        INPUT: None (reads current values from all form fields).
        OUTPUT: None. Runs the full inference pipeline on the single row currently entered in
            the form and displays the top-k decoded predictions, or an error message if any
            field is missing or contains a value unseen during training.
        """
        input_row = {col: self.field_widgets[col].get().strip() for col in FEATURE_COLUMNS}

        missing = [col for col, value in input_row.items() if value == '']
        if missing:
            messagebox.showwarning(
                "Missing fields",
                f"Please fill in all fields before predicting. Missing: {', '.join(missing)}"
            )
            return

        predictions = self._run_prediction(input_row)
        if predictions is not None:
            self._display_predictions(predictions)

    def _on_test_real_data(self):
        """
        INPUT: None.
        OUTPUT: None. Draws one random row from the held-out test sample, fills every form
            field with its feature values, runs the prediction pipeline on it, and displays the
            model's top-k predictions alongside the actual ('paramf') label for direct
            comparison.
        """
        test_samples = self.resources.get('test_samples')
        if test_samples is None or test_samples.empty:
            messagebox.showerror(
                "No test data",
                f"No test sample file was found at:\n{TEST_DATA_PATH}"
            )
            return

        sampled_row = test_samples.sample(n=1).iloc[0]
        input_row = {col: str(sampled_row[col]) for col in FEATURE_COLUMNS}
        real_value = str(sampled_row[TARGET_COLUMN])

        self._fill_fields(input_row)

        predictions = self._run_prediction(input_row)
        if predictions is not None:
            self._display_predictions(predictions, real_value=real_value)

    def _run_prediction(self, input_row):
        """
        INPUT: input_row (dict[str, str]): raw feature values for a single sample.
        OUTPUT: list[str] | None: the top-k decoded predicted classes, or None if an unseen
            categorical value prevented encoding (an error dialog is shown in that case).
        """
        try:
            return predict_top_k(
                input_row=input_row,
                feature_encoders=self.resources['feature_encoders'],
                base_models=self.resources['base_models'],
                meta_model=self.resources['meta_model'],
                meta_metadata=self.resources['meta_metadata'],
                target_encoder=self.resources['target_encoder'],
                k=TOP_K,
            )
        except UnseenLabelError as exc:
            messagebox.showerror("Unknown value", str(exc))
            return None

    def _display_predictions(self, predictions, real_value=None):
        """
        INPUT: predictions (list[str]): decoded predicted class labels, ranked by confidence.
               real_value (str | None): actual ground-truth label to display alongside the
                   predictions, when available (i.e. when the sample came from the held-out
                   test file); omitted for manually entered rows.
        OUTPUT: None. Writes the ranked predictions (and, if provided, the real value and
            whether it was recovered within the top-k) into the output text area.
        """
        self.output_text.configure(state='normal')
        self.output_text.delete('1.0', tk.END)

        if real_value is not None:
            hit_marker = "correct" if real_value in predictions else "not in top-k"
            self.output_text.insert(tk.END, f"Real value: {real_value}  ({hit_marker})\n\n")

        for rank, label in enumerate(predictions, start=1):
            self.output_text.insert(tk.END, f"{rank}. {label}\n")
        self.output_text.configure(state='disabled')


def main():
    """
    INPUT: None.
    OUTPUT: None. Loads all resources required for inference and starts the GUI event loop.
    """
    resources = load_all_resources(
        models_dir=MODELS_DIR,
        encoders_dir=ENCODERS_DIR,
        columns_dir=COLUMNS_DIR,
        autocomplete_columns=EDITABLE_COLUMNS,
        test_data_path=TEST_DATA_PATH,
    )
    app = PredictionApp(resources)
    app.mainloop()


if __name__ == '__main__':
    main()
