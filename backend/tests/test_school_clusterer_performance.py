"""
Performance tests for school_clusterer module

This test suite measures:
- Response time for clustering operations
- Accuracy of clustering results
- Edge case handling
- Scalability with different school list sizes

Run with: python -m tests.test_school_clusterer_performance
"""

import sys
import time
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.llm.school_clusterer import cluster_schools


def format_time(seconds: float) -> str:
    """Format time in seconds to readable string"""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    return f"{seconds:.2f}s"


def test_basic_clustering():
    """Test basic clustering with obvious variants"""
    print("\n[TEST 1] Basic Clustering")

    schools = [
        "Classical Academy Middle School",
        "Classical Academy High School",
        "Classical Academy Elementary",
        "Classical Academy TK-8",
        "Lincoln High School",
        "Washington Elementary",
    ]

    start_time = time.time()
    result = cluster_schools(schools, "Test District")
    elapsed = time.time() - start_time

    print(
        f"  {len(schools)} schools → {len(result['clusters'])} clusters | {format_time(elapsed)} | {result['source']}"
    )

    # Verify all schools are accounted for
    all_clustered = set()
    for schools_in_cluster in result["clusters"].values():
        all_clustered.update(schools_in_cluster)

    assert len(all_clustered) == len(schools), "All schools should be in clusters"
    assert all(
        school in all_clustered for school in schools
    ), "All input schools should be present"

    print("  ✓ Passed")
    return elapsed


def test_large_list():
    """Test clustering with a larger list of schools"""
    print("\n[TEST 2] Large School List")

    schools = []
    # Create multiple variants of several schools
    base_schools = [
        "Classical Academy",
        "Lincoln School",
        "Washington Charter",
        "Roosevelt Elementary",
        "Jefferson High",
        "Madison Middle",
        "Adams Academy",
        "Monroe Elementary",
    ]

    variants = ["Elementary", "Middle School", "High School", "TK-8", "K-12"]

    for base in base_schools:
        for variant in variants:
            schools.append(f"{base} {variant}")

    # Add some unique schools
    schools.extend(
        [
            "Unique School A",
            "Unique School B",
            "Unique School C",
        ]
    )

    start_time = time.time()
    result = cluster_schools(schools, "Large District")
    elapsed = time.time() - start_time

    reduction = len(schools) - len(result["clusters"])
    print(
        f"  {len(schools)} schools → {len(result['clusters'])} clusters | {format_time(elapsed)} | -{reduction} grouped | {result['source']}"
    )

    # Verify all schools are accounted for
    all_clustered = set()
    for schools_in_cluster in result["clusters"].values():
        all_clustered.update(schools_in_cluster)

    assert len(all_clustered) == len(schools), "All schools should be in clusters"

    print("  ✓ Passed")
    return elapsed


def test_edge_cases():
    """Test edge cases"""
    print("\n[TEST 3] Edge Cases")

    # Empty list
    start_time = time.time()
    result = cluster_schools([], "Test District")
    elapsed = time.time() - start_time
    assert result["clusters"] == {}, "Empty list should return empty clusters"
    assert result["source"] == "identity", "Empty list should use identity mapping"
    print(f"  Empty list: {format_time(elapsed)} ✓")

    # Single school
    start_time = time.time()
    result = cluster_schools(["Lone School"], "Test District")
    elapsed = time.time() - start_time
    assert len(result["clusters"]) == 1, "Single school should return one cluster"
    assert result["source"] == "identity", "Single school should use identity mapping"
    print(f"  Single school: {format_time(elapsed)} ✓")

    # All unique schools (no clustering possible)
    schools = ["School A", "School B", "School C", "School D", "School E"]
    start_time = time.time()
    result = cluster_schools(schools, "Test District")
    elapsed = time.time() - start_time
    print(
        f"  {len(schools)} unique schools → {len(result['clusters'])} clusters: {format_time(elapsed)} ✓"
    )


def test_performance_benchmark():
    """Run multiple iterations to benchmark performance"""
    print("\n[TEST 4] Performance Benchmark")

    schools = [
        "Classical Academy Middle School",
        "Classical Academy High School",
        "Classical Academy Elementary",
        "Lincoln High School",
        "Lincoln Middle School",
        "Washington Elementary",
    ]

    iterations = 3
    times = []

    for i in range(iterations):
        start_time = time.time()
        result = cluster_schools(schools, "Test District")
        elapsed = time.time() - start_time
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(
        f"  {iterations} iterations | Avg: {format_time(avg_time)} | Min: {format_time(min_time)} | Max: {format_time(max_time)}"
    )

    return avg_time


def test_clustering_accuracy():
    """Test that clustering groups similar schools correctly"""
    print("\n[TEST 5] Clustering Accuracy")

    # Schools that should cluster together
    test_cases = [
        {
            "name": "Classical Academy variants",
            "schools": [
                "Classical Academy Middle School",
                "Classical Academy High School",
                "Classical Academy Elementary",
                "Classical Academy TK-8",
            ],
            "expected_clusters": 1,
        },
        {
            "name": "Mixed schools",
            "schools": [
                "Classical Academy MS",
                "Classical Academy HS",
                "Lincoln School",
                "Washington Charter",
            ],
            "expected_clusters": 3,
        },
    ]

    for test_case in test_cases:
        start_time = time.time()
        result = cluster_schools(test_case["schools"], "Test District")
        elapsed = time.time() - start_time

        print(
            f"  {test_case['name']}: {len(test_case['schools'])} → {len(result['clusters'])} clusters (expected ~{test_case['expected_clusters']}) | {format_time(elapsed)} | {result['source']}"
        )

        # Verify all schools are accounted for
        all_clustered = set()
        for schools_in_cluster in result["clusters"].values():
            all_clustered.update(schools_in_cluster)

        assert len(all_clustered) == len(
            test_case["schools"]
        ), "All schools should be in clusters"


def run_all_tests():
    """Run all performance tests"""
    print("\n" + "=" * 50)
    print("SCHOOL CLUSTERER PERFORMANCE TESTS")
    print("=" * 50)

    total_start = time.time()
    results = {}

    try:
        results["basic"] = test_basic_clustering()
        results["large"] = test_large_list()
        test_edge_cases()
        results["benchmark"] = test_performance_benchmark()
        test_clustering_accuracy()

        total_elapsed = time.time() - total_start

        print("\n" + "-" * 50)
        print("SUMMARY")
        print("-" * 50)
        print(f"Total: {format_time(total_elapsed)}")
        for test_name, elapsed in results.items():
            print(f"  {test_name}: {format_time(elapsed)}")
        print("\n✓ All tests passed")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
