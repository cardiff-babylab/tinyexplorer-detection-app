#!/usr/bin/env python3
"""
Simple subprocess communication API for Electron-Python integration.
Communicates via stdin/stdout using JSON messages.
"""

import json
import sys
import os
import threading
import time
import signal
import atexit
from calc import calc as real_calc
from face_detection import FaceDetectionProcessor

class SubprocessAPI:
    def __init__(self):
        self.face_processor = None
        self.running = True
        self.progress_messages = []
        self.python_logs = []
        
        # Initialize face processor with callbacks
        self.face_processor = FaceDetectionProcessor(
            self.progress_callback,
            self.completion_callback
        )
        
        # Capture stdout for logging (but we'll use stderr for logs to avoid conflicts)
        self.setup_logging()
        
        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()
        
    def setup_logging(self):
        """Setup logging to stderr to avoid conflicts with stdout communication"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stderr
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False
            # Flush any pending output
            sys.stdout.flush()
            sys.stderr.flush()
            
        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def cleanup(self):
        """Cleanup resources on exit"""
        self.logger.info("Cleaning up resources...")
        if self.face_processor:
            try:
                self.face_processor.stop_processing()
            except:
                pass
        # Flush outputs
        sys.stdout.flush()
        sys.stderr.flush()
        
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
        self.logger.info(f"Processing completed: {data}")
        
        # Send completion event to frontend
        self.send_event({
            'type': 'completion',
            'data': data,
            'timestamp': time.time()
        })
        
    def send_response(self, response_data):
        """Send JSON response to stdout"""
        try:
            json_response = json.dumps(response_data)
            print(json_response, flush=True)
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            
    def send_event(self, event_data):
        """Send event to frontend"""
        self.send_response({
            'type': 'event',
            'event': event_data
        })
        
    def handle_command(self, command):
        """Handle incoming command from Electron"""
        try:
            cmd_type = command.get('type')
            data = command.get('data', {})
            
            if cmd_type == 'ping':
                return {'status': 'success', 'message': 'pong'}
                
            elif cmd_type == 'calc':
                result = real_calc(data.get('math', ''))
                return {'status': 'success', 'result': result}
                
            elif cmd_type == 'echo':
                return {'status': 'success', 'message': data.get('text', '')}
                
            elif cmd_type == 'get_models':
                try:
                    models = self.face_processor.get_available_models()
                    return {'status': 'success', 'models': models}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
            elif cmd_type == 'load_model':
                try:
                    model_path = data.get('model_path')
                    success = self.face_processor.load_model(model_path)
                    return {'status': 'success' if success else 'error', 'message': 'Model loaded' if success else 'Failed to load model'}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
            elif cmd_type == 'start_processing':
                try:
                    folder_path = data.get('folder_path')
                    confidence = data.get('confidence', 0.5)
                    model = data.get('model', 'yolov8n.pt')
                    save_results = data.get('save_results', False)
                    results_folder = data.get('results_folder')
                    
                    # Start processing in a separate thread
                    thread = threading.Thread(
                        target=self.face_processor.process_folder,
                        args=(folder_path, confidence, model, save_results, results_folder)
                    )
                    thread.start()
                    
                    return {'status': 'success', 'message': 'Processing started'}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
            elif cmd_type == 'stop_processing':
                try:
                    self.face_processor.stop_processing()
                    return {'status': 'success', 'message': 'Processing stopped'}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
            elif cmd_type == 'get_results':
                try:
                    results = self.face_processor.get_results()
                    return {'status': 'success', 'results': results}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
            elif cmd_type == 'get_status':
                try:
                    status = {
                        'is_processing': self.face_processor.is_processing,
                        'results_count': len(self.face_processor.results)
                    }
                    return {'status': 'success', 'status_info': status}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
            elif cmd_type == 'get_progress':
                try:
                    messages = self.progress_messages[-10:]  # Last 10 messages
                    return {'status': 'success', 'messages': messages}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
            elif cmd_type == 'get_logs':
                try:
                    logs = self.python_logs[-20:]  # Last 20 log messages
                    return {'status': 'success', 'logs': logs}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
            elif cmd_type == 'export_csv':
                try:
                    output_path = data.get('output_path')
                    results = self.face_processor.get_results()
                    success = self.face_processor.export_results_to_csv(results, output_path)
                    return {'status': 'success' if success else 'error', 'message': 'CSV exported' if success else 'Failed to export CSV'}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
            elif cmd_type == 'process_video':
                try:
                    video_path = data.get('video_path')
                    confidence = data.get('confidence', 0.5)
                    result_folder = data.get('result_folder')
                    
                    if not result_folder:
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        result_folder = os.path.join(os.getcwd(), f"video_processing_{timestamp}")
                        os.makedirs(result_folder, exist_ok=True)
                    
                    thread = threading.Thread(
                        target=self.face_processor.process_video,
                        args=(video_path, confidence, result_folder)
                    )
                    thread.start()
                    
                    return {'status': 'success', 'message': 'Video processing started'}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
            elif cmd_type == 'get_model_info':
                try:
                    info = {
                        'current_model': self.face_processor.current_model_path,
                        'model_type': self.face_processor.model_type,
                        'retinaface_available': 'RetinaFace' in self.face_processor.get_available_models()
                    }
                    return {'status': 'success', 'info': info}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
                    
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
        self.logger.info("Starting subprocess API...")
        
        # Send ready signal
        self.send_response({'type': 'ready', 'message': 'Python subprocess ready'})
        
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