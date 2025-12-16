#!/usr/bin/env python3
"""
Minimal subprocess communication API for testing Python connection.
"""

import json
import sys
import os
import threading
import time
from pathlib import Path

class MinimalSubprocessAPI:
    def __init__(self):
        self.running = True
        self.processing = False
        self.last_results = None
        
        # Setup logging to stderr to avoid conflicts with stdout communication
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
                
            elif cmd_type == 'get_info':
                return {
                    'status': 'success', 
                    'info': {
                        'python_version': sys.version,
                        'platform': sys.platform,
                        'cwd': os.getcwd(),
                        'script_location': __file__
                    }
                }
                
            elif cmd_type == 'get_models':
                # In packaged mode, inform user about limitations
                return {
                    'status': 'warning', 
                    'models': [
                        '⚠️ ML Libraries Not Bundled - Run from Source for Full Features',
                        'Note: Face detection models require 3.7GB+ of ML libraries',
                        'To use all models, run: npm start (development mode)'
                    ],
                    'message': 'The packaged app does not include ML libraries to keep size manageable. Run from source for full functionality.'
                }
                
            elif cmd_type == 'start_processing':
                try:
                    self.logger.info(f"Received start_processing command with data: {data}")
                    folder_path = data.get('folder_path', '')
                    confidence = data.get('confidence', 0.5)
                    model = data.get('model', 'yolov8n.pt')
                    results_folder = data.get('results_folder', '')
                    
                    self.logger.info(f"Processing parameters: folder_path='{folder_path}', model='{model}', confidence={confidence}")
                    
                    # Start processing in a separate thread
                    thread = threading.Thread(
                        target=self.process_images,
                        args=(folder_path, confidence, model, results_folder)
                    )
                    thread.start()
                    
                    return {
                        'status': 'success', 
                        'message': 'Processing started'
                    }
                except Exception as e:
                    self.logger.error(f"Error starting processing: {e}")
                    return {'status': 'error', 'message': str(e)}
                
            elif cmd_type == 'stop_processing':
                self.processing = False
                return {
                    'status': 'success',
                    'message': 'Processing stopped'
                }
                
            elif cmd_type == 'get_results':
                # Return stored results from last processing
                return {
                    'status': 'success',
                    'results': self.last_results or {
                        'total_images_processed': 0,
                        'total_faces_detected': 0,
                        'processing_time': 0.0,
                        'images': []
                    }
                }
                
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
        self.logger.info("Starting minimal subprocess API...")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        self.logger.info(f"Script location: {__file__}")
        
        # Send ready signal
        self.send_response({'type': 'ready', 'message': 'Python subprocess ready (minimal)'})
        
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
                
        self.logger.info("Minimal subprocess API shutting down...")
    
    def process_images(self, folder_path, confidence, model, results_folder):
        """Process images with mock face detection"""
        try:
            self.processing = True
            self.logger.info(f"Starting processing: {folder_path} with model {model}")
            
            # Send initial progress
            self.send_event({
                'type': 'progress',
                'data': f'🚀 Starting face detection with model {model}'
            })
            
            # Determine if it's a single file or folder
            path_obj = Path(folder_path)
            if path_obj.is_file():
                files = [path_obj]
            else:
                # Get all image files in folder
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                files = [f for f in path_obj.rglob('*') if f.suffix.lower() in image_extensions]
            
            total_files = len(files)
            self.send_event({
                'type': 'progress',
                'data': f'📁 Found {total_files} image(s) to process'
            })
            
            if total_files == 0:
                self.send_event({
                    'type': 'completion',
                    'data': {
                        'status': 'completed',
                        'total_images': 0,
                        'total_faces': 0,
                        'processing_time': 0.0
                    }
                })
                return
            
            # Process each file
            total_faces = 0
            start_time = time.time()
            processed_images = []
            
            for i, file_path in enumerate(files):
                if not self.processing:
                    break
                
                self.send_event({
                    'type': 'progress',
                    'data': f'🔍 Processing image {i+1}/{total_files}: {file_path.name}'
                })
                
                # Simulate processing time
                time.sleep(0.5)
                
                # Mock face detection (simulate finding faces)
                import random
                faces_found = random.randint(0, 3)  # Random 0-3 faces
                total_faces += faces_found
                
                progress_percent = ((i + 1) / total_files) * 100
                
                # Send completion event for this image
                self.send_event({
                    'type': 'completion',
                    'data': {
                        'status': 'image_completed',
                        'image_index': i + 1,
                        'total_images': total_files,
                        'detections_in_image': faces_found,
                        'progress_percent': progress_percent,
                        'file_path': str(file_path)
                    }
                })
                
                self.send_event({
                    'type': 'progress',
                    'data': f'✅ Image {i+1}/{total_files}: {faces_found} face(s) detected'
                })
                
                # Store image results
                processed_images.append({
                    'file_path': str(file_path),
                    'faces_detected': faces_found,
                    'processing_time': 0.5  # Mock processing time per image
                })
            
            processing_time = time.time() - start_time
            
            if self.processing:
                # Store results for get_results command
                self.last_results = {
                    'total_images_processed': len(processed_images),
                    'total_faces_detected': total_faces,
                    'processing_time': processing_time,
                    'images': processed_images
                }
                
                # Send final completion
                self.send_event({
                    'type': 'completion',
                    'data': {
                        'status': 'completed',
                        'total_images': total_files,
                        'total_faces': total_faces,
                        'processing_time': processing_time,
                        'results_folder': results_folder
                    }
                })
                
                self.send_event({
                    'type': 'progress',
                    'data': f'🎉 Processing completed! Found {total_faces} face(s) in {total_files} image(s) ({processing_time:.1f}s)'
                })
            else:
                self.send_event({
                    'type': 'progress',
                    'data': '⛔ Processing stopped by user'
                })
                
        except Exception as e:
            self.logger.error(f"Error in image processing: {e}")
            self.send_event({
                'type': 'progress',
                'data': f'❌ Error processing images: {str(e)}'
            })
        finally:
            self.processing = False

if __name__ == "__main__":
    api = MinimalSubprocessAPI()
    api.run()