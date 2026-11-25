"""``com_to`` argument parsing and |drone| configuration helpers for the ``R`` class.

Normalizes the ``com_to`` arguments accepted by every ``R`` data-acquisition
method and reads/writes per-|drone| configuration values.

.. codeauthor:: Darshan Patel <dp649@cornell.edu>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import ccatkidlib.io as io
import ccatkidlib.log as log
import ccatkidlib.utils as utils

if TYPE_CHECKING:
    from ccatkidlib.rfsoc.rfsoc_daq import R


# @function_timer
def group_args(com_to: list[str], *args: Any) -> dict[str, list[str]]:
    """Group |drone| ``com_to`` identifiers by their shared argument values.

    Zips the per-|drone| argument sequences into a comma-joined key and maps
    each unique key to the list of ``com_to`` identifiers sharing those values.

    Args:
        com_to (list[str]): |drone| identifiers, each in ``'Board.Drone'`` form.
        *args (Any): One sequence per argument, each aligned with ``com_to``.
    Returns:
        dict[str, list[str]]: Mapping of a comma-joined argument string to the
            list of ``com_to`` identifiers sharing those argument values.
    """
    arg_list = []
    for arg in zip(*args):
        arg = ", ".join([str(a) for a in list(arg)])
        arg_list.append(arg)

    sweep_dict = {}
    for com, arg in zip(com_to, arg_list):
        com_list = sweep_dict.setdefault(arg, [])
        com_list.append(com)

    return sweep_dict


def parse_args(R: R, com_to: list[str], arg: Any | list[Any]) -> list[Any] | None:
    """Normalize a single argument into a per-|drone| list.

    Args:
        R (R): The ``R`` data-acquisition instance.
        com_to (list[str]): |drone| identifiers the argument applies to.
        arg (Any | list[Any]): A single value (broadcast to every |drone|) or a
            list whose length matches ``com_to``.
    Returns:
        list[Any] | None: One argument per |drone|, or ``None`` if ``arg`` is a
            list whose length does not match ``com_to``.
    """

    args = None
    try:  # Check if argument is a list
        if isinstance(arg, str):
            raise TypeError
        if len(arg) == len(
            com_to
        ):  # Length of argument list should match number of drones
            args = arg
        else:
            log.log(
                "WARNING",
                f"{arg} is not a valid argument. Must be a single value or match the length of {com_to}!",
            )
            return None  # Return None if argument is invalid
    except TypeError:
        try:  # Assume argument is a single value and create a list of arugments with length equal to the number of drones
            args = [arg] * len(com_to)
        except:
            log.log(
                "WARNING",
                f"{arg} is not a valid argument. Must be a single value or match the length of {com_to}!",
            )
            return None  # Return None if argument is invalid
    return args  # Return parsed argument


def get_drone_args(R: R, com_to: list[str], key: str | list[str]) -> list[Any]:
    """Read a configuration value from each |drone|'s config file.

    Args:
        R (R): The ``R`` data-acquisition instance.
        com_to (list[str]): |drone| identifiers to read from.
        key (str | list[str]): Config key, or nested key path, to retrieve.
    Returns:
        list[Any]: The value of ``key`` from each |drone|'s configuration.
    """

    inds = [
        R.drone_list.index(com) for com in com_to
    ]  # Get list of indicies corresponding to drones in com_to
    return [
        utils.dict_get(R.drone_cfg[ind], key) for ind in inds
    ]  # Get dictionary value corresponding to key for each drone config file


def set_drone_args(
    R: R,
    com_to: list[str],
    key: str | list[str],
    args: list[Any],
) -> list[bool]:
    """Write a configuration value into each |drone|'s config file.

    Args:
        R (R): The ``R`` data-acquisition instance.
        com_to (list[str]): |drone| identifiers to write to.
        key (str | list[str]): Config key, or nested key path, to set.
        args (list[Any]): Value to set for each |drone|, aligned with ``com_to``.
    Returns:
        list[bool]: The return of ``io.edit_config`` for each |drone|.
    """

    # Iterate over drones and args
    rtn_list = []
    for com, arg in zip(com_to, args):
        ind = R.drone_list.index(com)  # Get index corresponding to drone in com_to
        rtn = io.edit_config(
            R.drone_cfg[ind], key, arg
        )  # Set config value of specified key
        rtn_list.append(rtn)
    return rtn_list


# @function_timer
def get_com_to(R: R, **kwargs: Any) -> tuple[list[str], list[str]]:
    """Resolve and set up the list of |drone| ``com_to`` identifiers.

    Defaults to the ``drone_list`` from the system config when no ``com_to`` keyword
    is given. A board-only identifier (e.g. ``'1'``) is expanded to all four of its
    drones. Every resolved |drone| is validated against the system-config drone list.

    Args:
        R (R): The ``R`` data-acquisition instance.
        **kwargs (Any): Accepts ``com_to`` (``str`` or ``list[str]``) and ``setup``
            (``bool``); remaining keywords are forwarded to ``R.setup_drones``.
    Returns:
        tuple[list[str], list[str]]: The sorted, de-duplicated |drone| ``com_to``
            list and the sorted list of boards they belong to.
    Raises:
        ValueError: If any requested |drone| is not in the system-config drone list.
    """

    # Set com_to to drone list specified in system config (make a copy so that it does not point to the R drone_list attribute)
    com_to = np.copy(R.drone_list).tolist()
    bids = R.board_list
    setup = True

    # Override com_to with that passed as key word argument (if any)
    for key, value in kwargs.items():
        if key == "com_to":
            com_to = value

            # If com_to is not a list, make it a list
            if isinstance(com_to, str):
                com_to = [com_to]

            # Get list of boards used
            bids = set()
            for com in com_to[::-1]:
                split_str = com.split(".")  # Split drone com_to into bid and drid
                bids.add(split_str[0])  # Add bid to set of board ids

                # Replace any board only com_to (e.g. '1') with bid.drid for all four drones
                if len(split_str) == 1:
                    com_to.remove(com)
                    for i in range(4):
                        com_to.append(com + f".{i + 1}")

            # Remove any duplicate entries and sort list of drones
            com_to = sorted(list(set(com_to)))

            # Check that all drones are in initialized drone list
            extra_drones = set(com_to) - set(
                R.drone_list
            )  # Get drones in com_to that are not in drone_list
            if len(extra_drones) > 0:
                raise ValueError(
                    f"The drones {sorted(list(extra_drones))} are not in system config drone list!"
                )
        elif key == "setup":
            setup = value

    # Set up drone with specified com_to list
    kwargs["restart"] = False
    if setup:
        R.setup_drones(**kwargs)

    # Return list of drones and list of boards in use
    return com_to, sorted(list(bids))
