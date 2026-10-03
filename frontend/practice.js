const API_URL ="https://signbridge-2-2ftb.onrender.com";

const params = new URLSearchParams(window.location.search);
const targetSign = params.get("sign") || "Hello";

const targetSignElement = document.getElementById("targetSign");
const targetMeaningElement = document.getElementById("targetMeaning");
const resultElement = document.getElementById("practiceResult");
const video = document.getElementById("practiceCamera");

const canvas = document.createElement("canvas");

const meanings = {
    "Hello": "नमस्कार",
    "Thank You": "धन्यवाद",
    "Yes": "हो",
    "No": "नाही",
    "Help": "मदत",
    "Stop": "थांबा",
    "Water": "पाणी",
    "Food": "अन्न",
    "Goodbye": "निरोप",
    "Emergency": "आपत्कालीन मदत",
    "Sorry": "माफ करा",
    "Please": "कृपया",
    "School": "शाळा",
    "Home": "घर",
    "Doctor": "डॉक्टर"
};

targetSignElement.textContent = "Practice: " + targetSign;

targetMeaningElement.textContent =
    "Target meaning: " + (meanings[targetSign] || targetSign);

let cameraStream = null;
let practiceRunning = false;
let checkingFrame = false;
let correctCount = 0;

function normalize(text) {
    return text
        .toLowerCase()
        .replace(/_/g, " ")
        .trim();
}

async function startPracticeCamera() {

    try {

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = cameraStream;

        practiceRunning = true;
        correctCount = 0;

        resultElement.textContent =
            "Show the sign: " + targetSign;

        recognizePracticeSign();

    } catch (error) {

        console.error(error);

        resultElement.textContent =
            "Unable to access camera.";

    }
}

async function recognizePracticeSign() {

    if (!practiceRunning || checkingFrame) {
        return;
    }

    if (video.readyState < 2) {
        setTimeout(recognizePracticeSign, 500);
        return;
    }

    checkingFrame = true;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvas.toBlob(async (blob) => {

        try {

            if (!blob) {
                return;
            }

            const formData = new FormData();

            formData.append(
                "image",
                blob,
                "practice.jpg"
            );

            const response = await fetch(
                `${API_URL}/predict`,
                {
                    method: "POST",
                    body: formData
                }
            );

            const data = await response.json();

            if (!data.prediction) {

                correctCount = 0;

                resultElement.textContent =
                    "✋ Hand not detected. Show your hand clearly.";

            } else {

                const detected = data.prediction;

                resultElement.textContent =
                    "Detected: " + data.meaning;

                if (
                    normalize(detected) ===
                    normalize(targetSign)
                ) {

                    correctCount++;

                    if (correctCount >= 2) {

                        resultElement.textContent =
                            "✅ Correct! " +
                            (meanings[targetSign] || targetSign);

                        correctCount = 0;

                    } else {

                        resultElement.textContent =
                            "✅ Good! Hold the sign...";

                    }

                } else {

                    correctCount = 0;

                    resultElement.textContent =
                        "❌ Detected: " +
                        data.meaning +
                        " — Try the " +
                        targetSign +
                        " sign.";

                }
            }

        } catch (error) {

            console.error(error);

            resultElement.textContent =
                "Could not connect to the AI backend.";

        } finally {

            checkingFrame = false;

            if (practiceRunning) {
                setTimeout(
                    recognizePracticeSign,
                    800
                );
            }

        }

    }, "image/jpeg");
}
