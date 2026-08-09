from trace_to_micro.state_diff import canonicalize_value, diff_snapshots, flatten_state


def test_flatten_state_preserves_nested_paths() -> None:
    state = {"device": {"data_enabled": False}, "lines": [{"status": "active"}]}

    assert flatten_state(state) == {
        "device.data_enabled": False,
        "lines.0.status": "active",
    }


def test_diff_snapshots_records_updates_and_insertions() -> None:
    before = {"user": {"device": {"data_enabled": False}}}
    after = {
        "user": {
            "device": {
                "data_enabled": True,
                "network_status": "connected",
            }
        }
    }

    changes = diff_snapshots(before, after)

    assert [change.path for change in changes] == [
        "user.device.data_enabled",
        "user.device.network_status",
    ]
    assert changes[0].before is False
    assert changes[0].after is True
    assert changes[1].before_present is False
    assert changes[1].after_present is True


def test_canonicalize_value_removes_entity_specific_arguments() -> None:
    arguments = {
        "customer_id": "C1001",
        "line_id": "L1002",
        "gb_amount": 2.0,
        "contact": {"phone_number": "555-123-2002"},
    }

    assert canonicalize_value(arguments) == {
        "contact": {"phone_number": "<phone_number>"},
        "customer_id": "<customer_id>",
        "gb_amount": 2.0,
        "line_id": "<line_id>",
    }
