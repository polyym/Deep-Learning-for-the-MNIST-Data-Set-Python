"""
Unit tests for the MNIST Neural Network Flask application.

Run with: pytest test_app.py -v
"""

import pytest
import json
import os
import sys

# Set test environment before importing app
os.environ['FLASK_DEBUG'] = 'false'
os.environ['LOG_LEVEL'] = 'WARNING'
os.environ['RATE_LIMIT_ENABLED'] = 'false'

from app import app, training_state, ValidationError, validate_int, validate_float, validate_choice, validate_digit


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def reset_state():
    """Reset training state before each test."""
    training_state._is_training = False
    training_state._progress = []
    training_state._network = None
    training_state._results = None
    yield


# =============================================================================
# Validation Function Tests
# =============================================================================

class TestValidateInt:
    """Tests for validate_int function."""

    def test_valid_int(self):
        assert validate_int(5, 'test') == 5

    def test_string_int(self):
        assert validate_int('10', 'test') == 10

    def test_default_value(self):
        assert validate_int(None, 'test', default=42) == 42

    def test_min_value(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_int(5, 'test', min_val=10)
        assert 'at least 10' in str(exc_info.value)

    def test_max_value(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_int(100, 'test', max_val=50)
        assert 'at most 50' in str(exc_info.value)

    def test_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_int('not a number', 'test')
        assert 'valid integer' in str(exc_info.value)

    def test_required(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_int(None, 'test')
        assert 'required' in str(exc_info.value)


class TestValidateFloat:
    """Tests for validate_float function."""

    def test_valid_float(self):
        assert validate_float(3.14, 'test') == 3.14

    def test_string_float(self):
        assert validate_float('2.5', 'test') == 2.5

    def test_int_to_float(self):
        assert validate_float(5, 'test') == 5.0

    def test_default_value(self):
        assert validate_float(None, 'test', default=1.0) == 1.0

    def test_min_value(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_float(0.5, 'test', min_val=1.0)
        assert 'at least 1.0' in str(exc_info.value)

    def test_max_value(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_float(100.0, 'test', max_val=50.0)
        assert 'at most 50.0' in str(exc_info.value)


class TestValidateChoice:
    """Tests for validate_choice function."""

    def test_valid_choice(self):
        assert validate_choice('cb', 'method', ['cb', 'uhb']) == 'cb'

    def test_default_value(self):
        assert validate_choice(None, 'method', ['cb', 'uhb'], default='cb') == 'cb'

    def test_invalid_choice(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_choice('invalid', 'method', ['cb', 'uhb'])
        assert 'must be one of' in str(exc_info.value)


class TestValidateDigit:
    """Tests for validate_digit function."""

    def test_valid_digit(self):
        for d in range(10):
            assert validate_digit(d, 'digit') == d

    def test_invalid_digit_negative(self):
        with pytest.raises(ValidationError):
            validate_digit(-1, 'digit')

    def test_invalid_digit_too_large(self):
        with pytest.raises(ValidationError):
            validate_digit(10, 'digit')


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_check(self, client):
        response = client.get('/api/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['is_training'] == False
        assert data['has_model'] == False
        assert data['has_results'] == False


class TestStatusEndpoint:
    """Tests for /api/status endpoint."""

    def test_status_initial(self, client):
        response = client.get('/api/status')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['is_training'] == False
        assert data['progress'] == []
        assert data['has_results'] == False


class TestTrainEndpoint:
    """Tests for /api/train endpoint."""

    def test_train_requires_json(self, client):
        response = client.post('/api/train', data='not json')
        assert response.status_code == 400

    def test_train_validation_error_epochs(self, client):
        response = client.post('/api/train',
                               data=json.dumps({'epochs': -1}),
                               content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'epochs' in data['error'].lower()

    def test_train_validation_error_learning_rate(self, client):
        response = client.post('/api/train',
                               data=json.dumps({'learningRate': 'invalid'}),
                               content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'learningrate' in data['error'].lower()

    def test_train_validation_error_backprop_method(self, client):
        response = client.post('/api/train',
                               data=json.dumps({'backpropMethod': 'invalid'}),
                               content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'backpropmethod' in data['error'].lower()

    def test_train_validation_error_duplicate_digits(self, client):
        response = client.post('/api/train',
                               data=json.dumps({
                                   'digitA': 1,
                                   'digitB': 1,
                                   'digitC': 2,
                                   'digitD': 3
                               }),
                               content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'unique' in data['error'].lower()

    def test_train_validation_error_invalid_digit(self, client):
        response = client.post('/api/train',
                               data=json.dumps({'digitA': 15}),
                               content_type='application/json')
        assert response.status_code == 400

    def test_train_starts_with_defaults(self, client):
        response = client.post('/api/train',
                               data=json.dumps({}),
                               content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Training started'

    def test_train_rejects_when_training(self, client):
        # Start first training
        client.post('/api/train',
                    data=json.dumps({}),
                    content_type='application/json')

        # Try to start second training
        response = client.post('/api/train',
                               data=json.dumps({}),
                               content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'already in progress' in data['error'].lower()


class TestResultsEndpoint:
    """Tests for /api/results endpoint."""

    def test_results_not_available(self, client):
        response = client.get('/api/results')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'no results' in data['error'].lower()


class TestPredictEndpoint:
    """Tests for /api/predict endpoint."""

    def test_predict_no_model(self, client):
        response = client.post('/api/predict',
                               data=json.dumps({'image': 'base64data'}),
                               content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'no trained model' in data['error'].lower()

    def test_predict_requires_json(self, client):
        response = client.post('/api/predict', data='not json')
        assert response.status_code == 400

    def test_predict_requires_image(self, client):
        # First we need a model (mock this scenario)
        training_state._network = True  # Mock network exists

        response = client.post('/api/predict',
                               data=json.dumps({}),
                               content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'no image' in data['error'].lower()


class TestSampleImagesEndpoint:
    """Tests for /api/sample_images endpoint."""

    def test_sample_images_no_model(self, client):
        response = client.get('/api/sample_images')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'no trained model' in data['error'].lower()


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestTrainingState:
    """Tests for TrainingState thread safety."""

    def test_is_training_property(self):
        assert training_state.is_training == False
        training_state.is_training = True
        assert training_state.is_training == True

    def test_progress_management(self):
        training_state.add_progress({'epoch': 1})
        training_state.add_progress({'epoch': 2})

        progress = training_state.get_progress(10)
        assert len(progress) == 2
        assert progress[0]['epoch'] == 1
        assert progress[1]['epoch'] == 2

    def test_progress_limit(self):
        # Add more than 100 entries
        for i in range(150):
            training_state.add_progress({'epoch': i})

        # Should only keep last 100
        assert len(training_state._progress) == 100
        assert training_state._progress[0]['epoch'] == 50

    def test_reset(self):
        training_state.add_progress({'epoch': 1})
        training_state._results = {'test': 'data'}

        training_state.reset()

        assert len(training_state._progress) == 0
        assert training_state._results is None


# =============================================================================
# Error Handler Tests
# =============================================================================

class TestErrorHandlers:
    """Tests for error handlers."""

    def test_404_error(self, client):
        response = client.get('/nonexistent')
        assert response.status_code == 404

    def test_request_too_large(self, client):
        # Create a large payload
        app.config['MAX_CONTENT_LENGTH'] = 100  # Very small limit for testing

        large_data = 'x' * 200
        response = client.post('/api/train',
                               data=large_data,
                               content_type='application/json')
        assert response.status_code == 413

        # Reset to default
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
