document.getElementById("upload-form").addEventListener("submit", function(event) {
    event.preventDefault();

    const formData = new FormData();
    const imageFile = document.getElementById("image").files[0];
    formData.append("image", imageFile);

    const captionText = document.getElementById("caption-text");
    const loading = document.getElementById("loading");
    const attentionMaps = document.getElementById("attention-maps");

    captionText.innerText = "";
    attentionMaps.innerHTML = "";
    loading.style.display = "block";

    fetch("/generate-caption", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            loading.style.display = "none";

            if (data.caption) {
                captionText.innerText = `Caption: ${data.caption}`;

                // Fetch and display attention maps
                for (let i = 0; i < data.attention_count; i++) {
                    fetch(`/generate-attention/${i}`, {
                        method: "POST",
                        body: formData
                    })
                        .then(response => response.blob())
                        .then(blob => {
                            const img = document.createElement("img");
                            img.src = URL.createObjectURL(blob);
                            img.alt = `Attention Map ${i}`;
                            attentionMaps.appendChild(img);
                        });
                }
            } else {
                captionText.innerText = "Error generating caption.";
            }
        })
        .catch(error => {
            loading.style.display = "none";
            captionText.innerText = "Error generating caption.";
            console.error("Error:", error);
        });
});
