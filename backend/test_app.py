import os
import io
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upload_no_file(client):
    response = client.post('/api/upload', data={})
    assert response.status_code == 400
    assert b'No image uploaded' in response.data

def test_upload_fake_image(client):
    # Crea una imagen falsa en memoria
    data = {
        'image': (io.BytesIO(b"fake image data"), 'test.jpg')
    }
    response = client.post('/api/upload', data=data, content_type='multipart/form-data')
    # Puede fallar la detección, pero debe responder correctamente
    assert response.status_code in (200, 404)

# Puedes añadir más tests para los otros endpoints simulando flujos completos 