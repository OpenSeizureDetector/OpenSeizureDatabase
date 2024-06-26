def appendUniqueEntriesToLst(inLst, addLst):
    """
    appendUniqueEntriesToLst _summary_

    _extended_summary_

    Args:
        inLst (_type_): input List (which is modified on exit)
        addLst (_type_): list of values to add into inLst

    Returns:
        int: number of unique values added to inLst
    """
    nAdded = 0
    for val in addLst:
        if (not val in inLst):
            inLst.append(val)
            nAdded += 1
    return nAdded


def removeEntriesFromLst(inLst, delLst):
    """
    removeEntriesFromLst _summary_

    _extended_summary_

    Args:
        inLst (_type_): input List (which is modified on exit)
        delLst (_type_): list of values to be removed from inLst

    Returns:
        int: number of unique values removed from inLst
    """
    nRemoved = 0
    for val in delLst:
        if (val in inLst):
            inLst.remove(val)
            nRemoved += 1
    return nRemoved
