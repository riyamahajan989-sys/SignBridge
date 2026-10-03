const API_URL ="https://signbridge-2-2ftb.onrender.com" ;
const video = document.getElementById("camera");
const canvas = document.createElement("canvas");
const resultText = document.getElementById("result");

let recognitionRunning = false;

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = stream;

        recognitionRunning = true;

        recognizeSign();

    } catch (error) {
        alert("Unable to access camera.");
        console.error(error);
    }
}


async function recognizeSign() {

    if (!recognitionRunning) {
        return;
    }

    if (video.readyState >= 2) {

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

            if (!blob) {
                return;
            }

            const formData = new FormData();

            formData.append(
                "image",
                blob,
                "camera.jpg"
            );

            try {

                const response = await fetch(
                    "http://127.0.0.1:5000/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );

                const data = await response.json();

                if (data.meaning) {
                    resultText.textContent =
                        "Meaning: " + data.meaning;
                }

            } catch (error) {

                console.error(
                    "Prediction error:",
                    error
                );

                resultText.textContent =
                    "Backend connection error";

            }

        }, "image/jpeg");
    }

    setTimeout(recognizeSign, 1000);
}
async function recognizeUploadedImage() {

    const imageInput = document.getElementById("signImage");
    const uploadResult = document.getElementById("uploadResult");

    if (!imageInput.files.length) {
        uploadResult.textContent = "Please select an image first.";
        return;
    }

    const file = imageInput.files[0];

    const formData = new FormData();

    formData.append("image", file);

    uploadResult.textContent = "Recognizing...";

    try {

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (data.meaning) {

            uploadResult.textContent =
                "Meaning: " + data.meaning;

        } else {

            uploadResult.textContent =
                "No sign detected.";

        }

    } catch (error) {

        console.error(error);

        uploadResult.textContent =
            "Could not connect to the backend.";

    }
}
