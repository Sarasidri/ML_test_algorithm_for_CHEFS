"""
Helpers behind the GUI's autocomplete behaviour: filtering a field's suggestion list as the
user types, and looking up the majority (producttype, productspec) pair associated with a given
productcomp value.
"""


def filter_choices(user_input, choices):
    """
    INPUT: user_input (str): text currently typed by the user in a combobox.
           choices (list[str]): full list of known values for that field.
    OUTPUT: list[str]: the subset of choices containing user_input as a case-insensitive
        substring; returns choices unchanged when user_input is empty.
    """
    if not user_input:
        return choices
    lowered_input = user_input.lower()
    return [choice for choice in choices if lowered_input in choice.lower()]


def get_auto_completed_fields(productcomp_value, completion_table):
    """
    INPUT: productcomp_value (str): value currently entered in the 'productcomp' field.
           completion_table (pandas.DataFrame): lookup table indexed by 'productcomp', with
               columns 'producttype' and 'productspec' (see io_utils.load_completion_table).
    OUTPUT: tuple[str, str] | tuple[None, None]: the majority-frequency (producttype,
        productspec) pair observed for productcomp_value in the training data, or (None, None)
        if productcomp_value is not present in the lookup table.
    """
    if productcomp_value not in completion_table.index:
        return None, None
    row = completion_table.loc[productcomp_value]
    return row['producttype'], row['productspec']
