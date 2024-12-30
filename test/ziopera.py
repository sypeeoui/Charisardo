from multiprocessing import Pool, freeze_support

with Pool() as p:
    p.map(print, range(10))

if __name__ == "__main__":
    freeze_support()