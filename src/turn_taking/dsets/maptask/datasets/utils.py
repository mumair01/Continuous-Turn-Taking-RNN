# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-31 11:55:01
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-31 11:55:26
import threading

# Instantiating a lock object
# This will be used to ensure that multiple parallel threads will not be able to run the same function at the same time
# in the @run_once decorator written below
__lock = threading.Lock()


def run_once(f):
    """
    Decorator to run a function only once.

    :param f: function to be run only once during execution time despite the number of calls
    :return: The original function with the params passed to it if it hasn't already been run before
    """

    def wrapper(*args, **kwargs):
        """
        The actual wrapper where the business logic to call the original function resides

        :param args:
        :param kwargs:
        :return: The original function unless the wrapper has been run already
        """
        if not wrapper.has_run:
            with __lock:
                if not wrapper.has_run:
                    wrapper.has_run = True
                    return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper
