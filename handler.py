import runpod

from worker import handler as worker_handler


def handler(job):
    return worker_handler(job)


runpod.serverless.start({"handler": handler})
