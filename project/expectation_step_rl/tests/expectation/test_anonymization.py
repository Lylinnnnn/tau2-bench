from expectation_step_rl.expectation.anonymization import consistently_anonymize


def test_consistent_anonymization_preserves_identity_relations() -> None:
    values = [
        {"order": "#W2378156", "user": "yusuf_rossi_9620"},
        {"argument": "#W2378156"},
        {"result": "owner=yusuf_rossi_9620 order=#Z9999999"},
    ]

    transformed, counts = consistently_anonymize(values)

    assert transformed[0]["order"] == transformed[1]["argument"]
    assert transformed[0]["order"] != transformed[2]["result"].split("order=")[1]
    assert transformed[0]["user"] in transformed[2]["result"]
    assert counts == {"HASH": 2, "USER": 1}
