import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np


class AudioProcessor:
    def __init__(self, max_workers=None):
        """
        Inițializează procesorul audio cu un număr specific de workeri.
        :param max_workers: Numărul maxim de threaduri (implicit: numărul de nuclee CPU)
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def process_chunks_parallel(self, data, chunk_size, process_func, **kwargs):
        """
        Procesează datele audio în paralel folosind multiple threaduri.
        :param data: Datele audio de procesat
        :param chunk_size: Dimensiunea fiecărui chunk
        :param process_func: Funcția de procesare pentru fiecare chunk
        :param kwargs: Argumente adiționale pentru funcția de procesare
        :return: Datele procesate
        """
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        futures = []

        # Trimitem chunkurile către workeri
        for chunk in chunks:
            future = self.executor.submit(process_func, chunk, **kwargs)
            futures.append(future)

        # Colectăm rezultatele
        processed_chunks = []
        for future in as_completed(futures):
            processed_chunks.append(future.result())

        # Sortăm chunkurile în ordinea originală
        return np.concatenate(processed_chunks)

    def shutdown(self):
        """
        Oprește executorul de threaduri.
        """
        self.executor.shutdown()