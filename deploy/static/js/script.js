document.getElementById("upload-form").addEventListener("submit", function(event) {
    event.preventDefault();

    var formData = new FormData();
    var imageFile = document.getElementById("image").files[0];
    formData.append("image", imageFile);

    fetch("/generate-caption", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.caption) {
            document.getElementById("caption-text").innerText = data.caption;
        } else {
            document.getElementById("caption-text").innerText = "Error generating caption.";
        }
    })
    .catch(error => {
        document.getElementById("caption-text").innerText = "Error generating caption.";
        console.error("Error:", error);
    });
});
