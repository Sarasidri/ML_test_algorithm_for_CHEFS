"""
Custom Tkinter widget(s) used by the prediction GUI: a Combobox that filters its drop-down
suggestions to those matching the text currently typed by the user.
"""

from tkinter import ttk

from utils.autocomplete_utils import filter_choices


class AutocompleteCombobox(ttk.Combobox):
    """
    A ttk.Combobox variant that narrows its drop-down list to the entries matching the text
    currently typed, instead of requiring an exact match against the full list of known values.
    """

    def set_completion_list(self, completion_list):
        """
        INPUT: completion_list (list[str]): full set of known values for this field.
        OUTPUT: None. Stores the list internally, initializes the widget's values, and starts
            listening for key-release events to drive the filtering.
        """
        self._completion_list = sorted(set(completion_list), key=str.lower)
        self['values'] = self._completion_list
        self.bind('<KeyRelease>', self._handle_keyrelease)

    def _handle_keyrelease(self, event):
        """
        INPUT: event (tkinter.Event): key-release event triggered by user typing.
        OUTPUT: None. Updates the widget's drop-down values in place, filtering them to those
            matching the text currently entered.
        """
        # Navigation and control keys should not trigger re-filtering.
        if event.keysym in ('Up', 'Down', 'Left', 'Right', 'Return', 'Escape', 'Tab'):
            return
        self['values'] = filter_choices(self.get(), self._completion_list)
