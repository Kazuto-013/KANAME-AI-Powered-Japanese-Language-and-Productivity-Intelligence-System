from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.config import Base, get_db
from backend.main import app


def test_uploaded_dataset_can_be_reloaded_by_id():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    test_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = test_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        csv_data = b"score,group,constant\n1,A,x\n2,A,x\n2,A,x\n4,B,x\n100,B,x\n"
        upload = client.post(
            "/api/data/upload",
            files={"file": ("quality.csv", BytesIO(csv_data), "text/csv")},
        )
        assert upload.status_code == 200, upload.text
        dataset_id = upload.json()["dataset_id"]
        assert upload.json()["source_stored"] is True

        profile = client.post(
            "/api/data/profile",
            data={"dataset_id": dataset_id},
        )
        assert profile.status_code == 200, profile.text
        body = profile.json()
        assert body["error"] is None
        assert body["profile"]["rows"] == 5
        assert body["profile"]["duplicate_rows"] == 1
        score = next(
            item for item in body["profile"]["profiled_columns"]
            if item["column"] == "score"
        )
        assert score["outlier_count"] == 1

        metadata = client.get(f"/api/data/datasets/{dataset_id}").json()["dataset"]
        assert len(metadata["sample_records"]) == 5

        correlation = client.post(
            "/api/data/correlation-matrix",
            data={"dataset_id": dataset_id, "method": "pearson"},
        ).json()
        assert correlation["error"] is None
        assert correlation["correlation"]["columns"] == ["score"]
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
