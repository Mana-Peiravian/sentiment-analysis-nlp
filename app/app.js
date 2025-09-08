function analyze() {
    let text = document.getElementById("inputText").value;
  
    // Example: fake response
    let result = "Neutral 😐";
    if (text.includes("love")) result = "Positive 🙂";
    if (text.includes("hate")) result = "Negative 🙁";
  
    document.getElementById("result").innerText = "Prediction: " + result;
  }
  