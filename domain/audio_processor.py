# import threading
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import multiprocessing
#
# import numpy as np
#
#
# class AudioProcessor:
#     def __init__(self, max_workers=None):
#         """
#         Inițializează procesorul audio cu un număr specific de workeri.
#         :param max_workers: Numărul maxim de threaduri (implicit: numărul de nuclee CPU)
#         """
#         self.max_workers = max_workers or multiprocessing.cpu_count()
#         self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
#
#     def process_chunks_parallel(self, data, chunk_size, process_func, **kwargs):
#         """
#         Procesează datele audio în paralel folosind multiple threaduri.
#         :param data: Datele audio de procesat
#         :param chunk_size: Dimensiunea fiecărui chunk
#         :param process_func: Funcția de procesare pentru fiecare chunk
#         :param kwargs: Argumente adiționale pentru funcția de procesare
#         :return: Datele procesate
#         """
#         # Împărțim datele în chunks
#         chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
#         total_chunks = len(chunks)
#
#         # Procesăm chunks în paralel
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             # Trimitem toate chunks-urile spre procesare și păstrăm referințele la futures
#             future_to_index = {
#                 executor.submit(process_func, chunk, **kwargs): i
#                 for i, chunk in enumerate(chunks)
#             }
#
#             # Inițializăm lista de rezultate cu None pentru a păstra ordinea
#             processed_chunks = [None] * total_chunks
#
#             # Colectăm rezultatele păstrând ordinea
#             for future in as_completed(future_to_index):
#                 index = future_to_index[future]
#                 try:
#                     processed_chunks[index] = future.result()
#                 except Exception as e:
#                     print(f"Eroare la procesarea chunk-ului {index}: {e}")
#                     # În caz de eroare, folosim chunk-ul original
#                     processed_chunks[index] = chunks[index]
#
#         # Combinăm chunks-urile procesate în ordinea corectă
#         return np.concatenate(processed_chunks)
#
#     def shutdown(self):
#         """
#         Oprește executorul de threaduri.
#         """
#         self.executor.shutdown()

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
        # Pentru înregistrări scurte, procesăm totul ca un singur chunk
        if len(data) <= chunk_size * 2:
            try:
                return process_func(data, **kwargs)
            except Exception as e:
                print(f"Eroare la procesarea înregistrării scurte: {e}")
                return data

        # Calculăm suprapunerea (10% din chunk_size)
        overlap = int(chunk_size * 0.1)

        # Împărțim datele în chunks cu suprapunere
        chunks = []
        chunk_indices = []
        for i in range(0, len(data), chunk_size - overlap):
            # Calculăm începutul și sfârșitul chunk-ului
            start = max(0, i - overlap) if i > 0 else 0
            end = min(len(data), i + chunk_size)

            # Extragem chunk-ul cu suprapunere
            chunk = data[start:end]
            chunks.append(chunk)
            chunk_indices.append((start, end))

        total_chunks = len(chunks)

        # Procesăm chunks în paralel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Trimitem toate chunks-urile spre procesare și păstrăm referințele la futures
            future_to_index = {
                executor.submit(process_func, chunk, **kwargs): i
                for i, chunk in enumerate(chunks)
            }

            # Inițializăm lista de rezultate cu None pentru a păstra ordinea
            processed_chunks = [None] * total_chunks

            # Colectăm rezultatele păstrând ordinea
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    processed_chunk = future.result()
                    # Asigurăm că chunk-ul procesat are aceeași lungime ca originalul
                    if len(processed_chunk) != len(chunks[index]):
                        # Dacă lungimea este diferită, ajustăm la lungimea originală
                        if len(processed_chunk) > len(chunks[index]):
                            processed_chunk = processed_chunk[:len(chunks[index])]
                        else:
                            # Dacă este mai scurt, extindem cu padding
                            padding = np.zeros(len(chunks[index]) - len(processed_chunk))
                            processed_chunk = np.concatenate([processed_chunk, padding])
                    processed_chunks[index] = processed_chunk
                except Exception as e:
                    print(f"Eroare la procesarea chunk-ului {index}: {e}")
                    # În caz de eroare, folosim chunk-ul original
                    processed_chunks[index] = chunks[index]

        # Combinăm chunks-urile procesate cu crossfade
        result = np.zeros(len(data))
        for i, (start, end) in enumerate(chunk_indices):
            chunk = processed_chunks[i]
            chunk_len = end - start

            # Aplicăm crossfade la început și sfârșitul chunk-ului
            if i > 0:  # Crossfade la început
                fade_in = np.linspace(0, 1, overlap)
                chunk[:overlap] *= fade_in
                result[start:start + overlap] += chunk[:overlap]

            if i < total_chunks - 1:  # Crossfade la sfârșit
                fade_out = np.linspace(1, 0, overlap)
                chunk[-overlap:] *= fade_out
                result[end - overlap:end] += chunk[-overlap:]

            # Adăugăm partea centrală a chunk-ului
            if i > 0 and i < total_chunks - 1:
                result[start + overlap:end - overlap] = chunk[overlap:-overlap]
            elif i == 0:
                result[start:end - overlap] = chunk[:-overlap]
            else:  # ultimul chunk
                result[start + overlap:end] = chunk[overlap:]

        return result

    def shutdown(self):
        """
        Oprește executorul de threaduri.
        """
        self.executor.shutdown()