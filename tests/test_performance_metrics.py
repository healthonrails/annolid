from annolid.realtime.perception import PerformanceMetrics


def test_performance_metrics_record_frame_none_handling():
    metrics = PerformanceMetrics()

    # Normal case
    metrics.record_frame(0.01, 5)
    assert metrics.detection_count == 5
    assert metrics.frame_count == 1
    assert len(metrics.latency_history) == 1

    # None handling case (simulating the bug fix)
    metrics.record_frame(0.02, None)
    assert metrics.detection_count == 5  # Should remain 5 (0 added)
    assert metrics.frame_count == 2
    assert len(metrics.latency_history) == 2


def test_performance_metrics_record_frame_normal():
    metrics = PerformanceMetrics()
    metrics.record_frame(0.01, 10)
    assert metrics.detection_count == 10
    assert metrics.frame_count == 1
