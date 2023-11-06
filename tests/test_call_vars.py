import pytest
import sys
import threading
import time
import asyncio
import uuid

from nr_openai_observability.call_vars import set_conversation_id, get_conversation_id

failures = 0

def test_set_conversation_id():
    global failures

    def set_conversation_id_thread():
        global failures
        test_id = str(uuid.uuid4())
        #Check that each new call gets a fresh context
        if get_conversation_id() is not None:
            failures += 1
        set_conversation_id(test_id)
        time.sleep(0.001)
        #check that context hasn't been polluted with a different thread
        if test_id != get_conversation_id():
            failures += 1

    # Greatly improve the chance of an operation being interrupted
    # by thread switch.
    try:
        sys.setswitchinterval(1e-12)
    except AttributeError:
        # Python 2 compatible
        sys.setcheckinterval(1)

    threads = []
    for _ in range(1000):
        t = threading.Thread(target=set_conversation_id_thread)
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()

    assert failures == 0