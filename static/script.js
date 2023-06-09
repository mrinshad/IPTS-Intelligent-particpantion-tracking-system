const progressBar = document.getElementById('progress-bar');
const status = document.getElementById('status');
const formData = new FormData(form);
const nameValue = nameInput.value;
formData.append('name', nameValue);

function startTraining() {
  progressBar.style.width = '0%';
  progressBar.classList.remove('hidden');
  status.textContent = 'Please wait while we take some pictures...';
  status.classList.remove('hidden');

  fetch('/train', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: formData,
  })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        updateProgressBar();
      } else {
        // Handle error response
        progressBar.classList.add('hidden');
        status.textContent = 'Training failed. Please try again.';
      }
    })
    .catch(error => {
      // Handle network error
      progressBar.classList.add('hidden');
      status.textContent = 'An error occurred. Please try again later.';
    });
}

function updateProgressBar() {
  let width = 0;
  const intervalId = setInterval(() => {
    width += 10;
    progressBar.style.width = `${width}%`;
    if (width >= 100) {
      clearInterval(intervalId);
      progressBar.classList.add('hidden');
      status.textContent = 'Training complete';
      // Reload the train.html page
      setTimeout(() => {
        window.location.reload();
      }, 2000); // Adjust the delay as needed
    }
  }, 500);
  // closeAndOpenTrainPage();

}

document.getElementById('train-form').addEventListener('submit', (e) => {
  e.preventDefault();
  startTraining();
});

// Inside your script.js file

// // Function to reload the train.html page
// function reloadTrainPage() {
//   window.location.reload(); // Reload the current page
// }

// // Function to close the train.html page and open it again
// function closeAndOpenTrainPage() {
//   window.location.href = '/train'; // Navigate to the train.html page
// }

// Example usage
// Call the reloadTrainPage function to reload the train.html page
// reloadTrainPage();

// Or call the closeAndOpenTrainPage function to close and open the train.html page
// closeAndOpenTrainPage();
