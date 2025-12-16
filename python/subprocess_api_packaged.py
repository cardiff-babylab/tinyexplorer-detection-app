#!/usr/bin/env python3
"""
Subprocess communication API for packaged Electron app.
This version handles missing dependencies gracefully.
"""

import json
import sys
import os
import threading
import time

# Try to import face detection, but fall back to minimal if not available
try:
    from face_detection import FaceDetectionProcessor
    FACE_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Face detection not available in packaged mode: {e}", file=sys.stderr)
    FACE_DETECTION_AVAILABLE = False

class SubprocessAPI:
    def __init__(self):
        self.face_processor = None
        self.running = True
        self.progress_messages = []
        self.python_logs = []
        
        # Initialize face processor only if available
        if FACE_DETECTION_AVAILABLE:
            try:
                self.face_processor = FaceDetectionProcessor(
                    self.progress_callback,
                    self.completion_callback
                )
            except Exception as e:
                print(f"Could not initialize FaceDetectionProcessor: {e}", file=sys.stderr)
                self.face_processor = None
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging to stderr to avoid conflicts with stdout communication"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stderr
        )
        self.logger = logging.getLogger(__name__)
        
    def progress_callback(self, message):
        """Called when processing progress updates"""
        self.progress_messages.append(message)
        self.python_logs.append(f"Progress: {message}")
        self.logger.info(f"Progress: {message}")
        
        # Send progress event to frontend
        self.send_event({
            'type': 'progress',
            'data': message,
            'timestamp': time.time()
        })
        
    def completion_callback(self, data):
        """Called when processing completes"""
        self.logger.info(f"Completion: {data}")
        
        # Send completion event to frontend
        self.send_event({
            'type': 'completion',
            'data': data
        })
        
    def send_response(self, response_data):
        """Send JSON response to stdout"""
        try:
            json_response = json.dumps(response_data)
            print(json_response, flush=True)
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            
    def send_event(self, event_data):
        """Send event to Electron main process"""
        try:
            event_message = {
                'type': 'event',
                'event': event_data
            }
            json_response = json.dumps(event_message)
            print(json_response, flush=True)
        except Exception as e:
            self.logger.error(f"Error sending event: {e}")
            
    def handle_command(self, command):
        """Handle incoming command from Electron"""
        try:
            cmd_type = command.get('type')
            data = command.get('data', {})
            
            if cmd_type == 'ping':
                return {'status': 'success', 'message': 'pong'}
                
            elif cmd_type == 'echo':
                return {'status': 'success', 'message': data.get('text', '')}
                
            elif cmd_type == 'get_models':
                if self.face_processor:
                    try:
                        models = self.face_processor.get_available_models()
                        return {'status': 'success', 'models': models}
                    except Exception as e:
                        return {'status': 'error', 'message': f'Error getting models: {str(e)}'}
                else:
                    # Return minimal model list for packaged app without ML dependencies
                    return {
                        'status': 'warning', 
                        'models': ['Basic Detection (Not Available - ML packages not bundled)'],
                        'message': 'Face detection requires ML packages that are not included in the packaged app. Please run from source for full functionality.'
                    }
                    
            elif cmd_type == 'start_processing':
                if not self.face_processor:
                    return {
                        'status': 'error',
                        'message': 'Face detection is not available in this packaged build. ML dependencies are not bundled to keep app size manageable.'
                    }
                
                # Original processing logic here...
                folder_path = data.get('folder_path', '')
                confidence = data.get('confidence', 0.5)
                model = data.get('model', 'yolov8n.pt')
                save_results = data.get('save_results', False)
                results_folder = data.get('results_folder', '')
                
                # Start processing in a separate thread
                thread = threading.Thread(
                    target=self.face_processor.process_folder,
                    args=(folder_path, confidence, model, save_results, results_folder)
                )
                thread.start()
                
                return {'status': 'success', 'message': 'Processing started'}
                
            elif cmd_type == 'stop_processing':
                if self.face_processor:
                    self.face_processor.stop_processing()
                return {'status': 'success', 'message': 'Processing stopped'}
                
            elif cmd_type == 'get_results':
                if self.face_processor:
                    results = self.face_processor.get_results()
                    return {'status': 'success', 'results': results}
                else:
                    return {'status': 'error', 'results': [], 'message': 'Face detection not available'}
                    
            elif cmd_type == 'exit':
                self.running = False
                return {'status': 'success', 'message': 'Exiting...'}
                
            else:
                return {'status': 'error', 'message': f'Unknown command type: {cmd_type}'}
                
        except Exception as e:
            self.logger.error(f"Error handling command: {e}")
            return {'status': 'error', 'message': str(e)}
            
    def run(self):
        """Main subprocess loop - read commands from stdin and send responses to stdout"""
        self.logger.info("Starting subprocess API (packaged mode)...")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Face detection available: {FACE_DETECTION_AVAILABLE}")
        
        # Send ready signal
        self.send_response({
            'type': 'ready', 
            'message': 'Python subprocess ready',
            'face_detection_available': FACE_DETECTION_AVAILABLE
        })
        
        while self.running:
            try:
                # Read line from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                # Parse JSON command
                try:
                    command = json.loads(line)
                except json.JSONDecodeError as e:
                    self.send_response({
                        'type': 'error',
                        'message': f'Invalid JSON: {e}'
                    })
                    continue
                    
                # Handle command
                response = self.handle_command(command)
                
                # Send response with command ID if present
                response_data = {
                    'type': 'response',
                    'response': response
                }
                if 'id' in command:
                    response_data['id'] = command['id']
                
                self.send_response(response_data)
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.send_response({
                    'type': 'error',
                    'message': str(e)
                })
                
        self.logger.info("Subprocess API shutting down...")

if __name__ == "__main__":
    api = SubprocessAPI()
    api.run()