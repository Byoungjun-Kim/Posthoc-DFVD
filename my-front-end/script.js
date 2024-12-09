document.getElementById("detectBtn").addEventListener("click", async () => {
    const videoLink = document.getElementById("videoLink").value;
  
    if (!videoLink) {
      alert("Please enter a valid YouTube link");
      return;
    }

    const videoIdMatch = videoLink.match(/(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([\w-]+)/) 
                        || videoLink.match(/(?:https?:\/\/)?(?:www\.)?youtu\.be\/([\w-]+)/);
    
    if (videoIdMatch) {
        const videoId = videoIdMatch[1];
        const embedUrl = `https://www.youtube.com/embed/${videoId}`;
        
        videoContainer.innerHTML = `<iframe width="560" height="315" src="${embedUrl}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>`;
    } else {
        videoContainer.innerHTML = `<p style="color: red;">Invalid YouTube link. Please try again.</p>`;
    }
  
    const resultElement = document.getElementById("result");
    resultElement.innerHTML = "Analyzing...";
  
    try {
      // 替换为你的后端 API 地址
      const response = await fetch("http://127.0.0.1:8080/fake_prob", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: videoLink }),
      });
  
      const data = await response.json();
      if (response.ok) {
        resultElement.innerHTML = `<strong>Fake probability:</strong> ${(data.fake_prob * 100).toFixed(2)}%`;
      } else {
        resultElement.innerHTML = `<strong>Error:</strong> ${data.message}`;
      }
    } catch (error) {
      console.error("Error:", error);
      resultElement.innerHTML = "An error occurred while processing the request.";
    }
  });
  