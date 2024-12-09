document.getElementById("experiment-form").addEventListener("submit", async function(event) {
    event.preventDefault();  // Prevent form submission

    const queryType = document.getElementById("query_type").value;
    const textQuery = document.getElementById("text_query").value;
    const imageQueryInput = document.getElementById("image_query").files[0];
    const QueryWeight = parseFloat(document.getElementById("QueryWeight").value);
    const PCA = document.getElementById("PCA").value;

    // Validation checks
    if (queryType === "Text Query" && textQuery === "") {
        alert("Please enter a text query.");
        return;
    }

    if (queryType === "Image Query" && !imageQueryInput) {
        alert("Please upload an image.");
        return;
    }

    // Prepare form data
    const formData = new FormData();
    if (queryType === "Text Query") {
        formData.append('text_query', textQuery);
    } else if (queryType === "Image Query") {
        formData.append('image_query', imageQueryInput);
    }

    formData.append('query_type', queryType);
    formData.append('QueryWeight', QueryWeight);

    // Send the query to the Flask app
    fetch("/run_experiment", {
        method: "POST",
        body: JSON.stringify({
            query_type: queryType,
            text_query: textQuery,
            image_query: imageQueryInput ? imageQueryInput.name : null,
            QueryWeight: QueryWeight,
            PCA: PCA
        }),
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        // Display images and similarity scores
        const resultsDiv = document.getElementById("results");
        const imagesDiv = document.getElementById("images");
        imagesDiv.innerHTML = "";

        data.forEach(result => {
            const imgElement = document.createElement("img");
            imgElement.src = result.image_path;
            imgElement.alt = "Similar Image";
            imgElement.width = 200;
            const similarityElement = document.createElement("p");
            similarityElement.innerText = `Similarity Score: ${result.similarity_score.toFixed(3)}`;
            imagesDiv.appendChild(imgElement);
            imagesDiv.appendChild(similarityElement);
        });

        resultsDiv.style.display = "block";
    })
    .catch(error => {
        console.error("Error running experiment:", error);
        alert("An error occurred while running the experiment.");
    });
});
