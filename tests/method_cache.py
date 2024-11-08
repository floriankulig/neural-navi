import time
import datetime
from timeit import timeit


def demonstrate_lookup_difference():
    """Demonstriert den Unterschied zwischen direktem und globalem Zugriff"""

    # Version 1: Ohne Caching
    def without_caching():
        for _ in range(1000000):
            current_time = time.time()  # Lookup in global namespace

    # Version 2: Mit Caching
    def with_caching():
        time_func = time.time  # Lokale Referenz
        for _ in range(1000000):
            current_time = time_func()  # Lookup in lokaler namespace

    # Zeitmessung
    time_without = timeit(without_caching, number=1)
    time_with = timeit(with_caching, number=1)

    return time_without, time_with


def explain_namespace_lookup():
    """Erklärt wie Python Namen auflöst"""

    def example_function():
        x = 1  # Lokale Variable

        def inner_function():
            # Python sucht Namen in dieser Reihenfolge:
            # 1. Lokaler Scope (L)
            # 2. Umgebender Scope (E)
            # 3. Globaler Scope (G)
            # 4. Built-in Scope (B)
            return x  # Findet x im umgebenden Scope

        return inner_function()

    return example_function()


def benchmark_different_approaches():
    """Vergleicht verschiedene Zugriffsmethoden"""

    ITERATIONS = 1000000

    def test_direct():
        for _ in range(ITERATIONS):
            t = time.time()

    def test_cached():
        time_func = time.time
        for _ in range(ITERATIONS):
            t = time_func()

    def test_method():
        class Timer:
            def get_time(self):
                return time.time()

        timer = Timer()
        for _ in range(ITERATIONS):
            t = timer.get_time()

    results = {
        "Direkter Zugriff": timeit(test_direct, number=1),
        "Gecachte Funktion": timeit(test_cached, number=1),
        "Methodenaufruf": timeit(test_method, number=1),
    }

    return results


if __name__ == "__main__":
    # Demonstriere Lookup-Unterschiede
    time_without, time_with = demonstrate_lookup_difference()
    print(f"Zeit ohne Caching: {time_without:.4f} Sekunden")
    print(f"Zeit mit Caching: {time_with:.4f} Sekunden")
    print(f"Verbesserung: {((time_without - time_with) / time_without * 100):.1f}%")

    # Benchmark verschiedener Ansätze
    print("\nBenchmark verschiedener Zugriffsmethoden:")
    results = benchmark_different_approaches()
    for method, duration in results.items():
        print(f"{method}: {duration:.4f} Sekunden")
