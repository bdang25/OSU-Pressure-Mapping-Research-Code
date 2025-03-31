// script.js

// 1) Preview the selected 6×6 file in the left image container
document.getElementById('image').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
      // Display the file name below the 6×6 image
      document.getElementById('input-filename').textContent = file.name;
  
      // Use FileReader to show an immediate preview of the selected file
      const reader = new FileReader();
      reader.onload = function(event) {
        const inputImg = document.getElementById('input-image');
        inputImg.src = event.target.result;
        inputImg.style.display = 'block';
      };
      reader.readAsDataURL(file);
    }
  });
  
  // 2) Handle form submission to the Flask backend
  document.getElementById('upload-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
  
    fetch('/generate', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      if (data.generated_image_url) {
        // Show the 32×32 image
        const genImg = document.getElementById('generated-image');
        genImg.src = data.generated_image_url;
        genImg.style.display = 'block';
  
        // If your Flask route also returns a filename, show that too
        if (data.generated_image_filename) {
          document.getElementById('generated-image-filename').textContent = data.generated_image_filename;
        } else {
          // If not, parse it from the URL or show a generic label
          const urlParts = data.generated_image_url.split('/');
          const fileName = urlParts[urlParts.length - 1];
          document.getElementById('generated-image-filename').textContent = fileName;
        }
      } else {
        alert("Error: " + (data.error || "No generated_image_url returned."));
      }
    })
    .catch(err => {
      console.error(err);
      alert("An error occurred while generating the 32×32 image.");
    });
  });
  