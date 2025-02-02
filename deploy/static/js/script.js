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
                    const data = await response.json();
                    return {
                        index: index,
                        data: data
                    };
                }
            } catch (error) {
                console.error(`Error loading attention map ${index}:`, error);
                return null;
            }
        };

        const displayAttentionMap = (mapData) => {
            const { data } = mapData;
            // Create container for this attention map
            const mapContainer = document.createElement("div");
            mapContainer.className = "attention-map-container";
            mapContainer.style.margin = "20px";
            mapContainer.style.display = "inline-block";
            
            // Create and set up image from Base64 data
            const img = document.createElement("img");
            img.src = `data:image/png;base64,${data.image_base64}`;
            img.alt = `Attention Map for: ${data.word}`;
            img.style.maxWidth = "400px";
            img.style.margin = "10px";
            
            // Create word label
            const label = document.createElement("div");
            label.textContent = `Word: ${data.word}`;
            label.style.color = "white";
            label.style.marginTop = "5px";
            
            // Add elements to container
            mapContainer.appendChild(img);
            mapContainer.appendChild(label);
            
            // Add container to attention maps section
            attentionMaps.appendChild(mapContainer);
        };

        // Load all attention maps
        const loadAndDisplayMaps = async (count) => {
            try {
                // Load all maps simultaneously but maintain order
                const results = await Promise.all(
                    [...Array(count)].map((_, i) => loadAttentionMap(i))
                );
                
                // Filter out any failed loads and sort by index
                const validResults = results
                    .filter(result => result !== null)
                    .sort((a, b) => a.index - b.index);
                
                // Display maps in order
                validResults.forEach(displayAttentionMap);
            } catch (error) {
                console.error("Error loading attention maps:", error);
            } finally {
                loading.style.display = "none";
            }
        };

        if (window.attentionCount) {
            loadAndDisplayMaps(window.attentionCount);
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
                        loadAndDisplayMaps(data.attention_count);
                    }
                })
                .catch(error => {
                    console.error("Error:", error);
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
