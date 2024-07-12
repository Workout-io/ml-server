from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from exercises.Squat2 import process_frame_squatjump
from exercises.DumbelCurl import process_frame_curl
from exercises.Lunges import process_frame_lunge
from exercises.Plank import process_frame_plank
from exercises.pushup_main import process_frame_pushup
from exercises.Situp import process_frame_situp

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Dictionary mapping exercise names to their respective processing functions
exercise_functions = {
    "Push Up": process_frame_pushup,
    "Squat": process_frame_squatjump,
    "Sit Up": process_frame_situp,
    "Lunges": process_frame_lunge,
    "Dumbell Curl": process_frame_curl,
    "Plank": process_frame_plank,
}


@app.route("/", methods=["GET"])
def index():
    return jsonify("Hello API"), 200


@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    exercise_name = request.form.get("exercise_name")
    if not exercise_name:
        return jsonify({"error": "Exercise name not provided"}), 400

    if exercise_name not in exercise_functions:
        return jsonify({"error": f"Unsupported exercise name: {exercise_name}"}), 400

    if file and file.filename.rsplit(".", 1)[1].lower() in {"mp4", "avi"}:
        # Save the uploaded file
        filename = file.filename
        file_path = os.path.join("tmp/videos", filename)
        file.save(file_path)

        # Call the corresponding exercise detection function
        detect_function = exercise_functions[exercise_name]
        results = detect_function(file_path)

        return jsonify(results), 200
    else:
        return (
            jsonify(
                {
                    "error": "Unsupported file format, please upload a video file (MP4 or AVI)"
                }
            ),
            400,
        )


if __name__ == "__main__":
    app.run(debug=True)
