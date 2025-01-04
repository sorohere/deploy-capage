document.getElementById("upload-form").addEventListener("submit", function (event) {
    event.preventDefault();
    
    const clickedButton = event.submitter;
    const formData = new FormData();
    const imageFile = document.getElementById("image").files[0];
    formData.append("image", imageFile);

    const captionText = document.getElementById("caption-text");
    const loading = document.getElementById("loading");
    const attentionMaps = document.getElementById("attention-maps");

    loading.style.display = "block";

    if (clickedButton.textContent === "getCaption") {
        // Fetch the caption without clearing previous attention maps
        fetch("/generate-caption", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                loading.style.display = "none";

                if (data.caption) {
                    captionText.innerText = `Caption: ${data.caption}`;
                    
                    // Store attention count for later use
                    window.attentionCount = data.attention_count;
                } else {
                    captionText.innerText = "Error generating caption.";
                }
            })
            .catch((error) => {
                loading.style.display = "none";
                captionText.innerText = "Error generating caption.";
                console.error("Error:", error);
            });
    } else if (clickedButton.textContent === "getAttention") {
        // Clear previous attention maps
        attentionMaps.innerHTML = "";
        
        // Create and load attention maps
        const loadAttentionMap = async (index) => {
            try {
                const response = await fetch(`/generate-attention/${index}`, {
                    method: "POST",
                    body: formData,
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const img = document.createElement("img");
                    img.src = URL.createObjectURL(blob);
                    img.alt = `Attention Map ${index + 1}`;
                    img.style.margin = "10px";
                    attentionMaps.appendChild(img);
                }
            } catch (error) {
                console.error(`Error loading attention map ${index}:`, error);
            }
        };

        // Load all attention maps
        if (window.attentionCount) {
            Promise.all([...Array(window.attentionCount)].map((_, i) => loadAttentionMap(i)))
                .finally(() => {
                    loading.style.display = "none";
                });
        } else {
            // If attention count is not available, fetch caption first
            fetch("/generate-caption", {
                method: "POST",
                body: formData,
            })
                .then((response) => response.json())
                .then((data) => {
                    if (data.attention_count) {
                        window.attentionCount = data.attention_count;
                        return Promise.all([...Array(data.attention_count)].map((_, i) => loadAttentionMap(i)));
                    }
                })
                .finally(() => {
                    loading.style.display = "none";
                });
        }
    }
});

// Add file input change listener to display selected image
document.getElementById("image").addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.createElement("img");
            img.src = e.target.result;
            img.style.maxWidth = "400px";
            img.style.marginTop = "20px";
            
            // Remove previous preview if exists
            const existingPreview = document.getElementById("image-preview");
            if (existingPreview) {
                existingPreview.remove();
            }
            
            // Add new preview
            const preview = document.createElement("div");
            preview.id = "image-preview";
            preview.appendChild(img);
            document.querySelector(".container").insertBefore(preview, document.getElementById("caption-text"));
        };
        reader.readAsDataURL(file);
    }
});
