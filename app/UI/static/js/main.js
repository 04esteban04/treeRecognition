document.addEventListener("DOMContentLoaded", () => {
	
	// Elements
	const messageBox = document.getElementById("message");

	const fileForm = document.getElementById("fileUploadFormIndividual");
	const fileFormBulk = document.getElementById("fileUploadFormBulk");
	const fileInput = document.getElementById("file");

	const defaultWrapper = document.getElementById("processDefaultBtnWrapper");
	const processDefaultBtn = document.getElementById("processDefaultDatasetBtn");
		
	const btnIndividualContainer = document.getElementById("Individual");
	const btnBulkContainer = document.getElementById("In-bulk");
	const btnDefaultContainer = document.getElementById("Default");
	
	const btnIndividual = document.getElementById("button-individual");
	const btnBulk = document.getElementById("button-bulk");
	const btnDefault = document.getElementById("button-default");


	function showMessage(msg, isError = false) {
		messageBox.textContent = msg;
		messageBox.classList.remove("d-none", "alert-info", "alert-danger");
		messageBox.classList.add(isError ? "alert-danger" : "alert-info");
	}

	// Handle file upload form submission
	function handleFileFormSubmit(formElement, isBulk = false) {
		formElement.addEventListener("submit", async (e) => {
			e.preventDefault();

			const fileInput = formElement.querySelector("input[type='file']");
			const file = fileInput?.files[0];

			if (!file) {
				showMessage("Please select a file to upload.", true);
				return;
			}

			const formData = new FormData();
			formData.append("file", file);

			let endpoint = isBulk ? "/process/batch" : "/process/image";

			try {
				const response = await fetch(endpoint, {
					method: "POST",
					body: formData,
				});

				const result = await response.json();
				if (response.ok) {
					showMessage(result.message || "File processed successfully.");
				} else {
					showMessage(result.error || "An error occurred during processing.", true);
				}
			} catch (error) {
				showMessage("Failed to process file: " + error.message, true);
			}
		});
	}

	// Reset all modes and hide forms
	function resetMode() {
		fileForm.classList.add("d-none");
		fileFormBulk.classList.add("d-none");
		defaultWrapper.classList.add("d-none");
		fileInput.value = "";
		messageBox.classList.add("d-none");
	}

	// Tab navigation from main page buttons
	document.getElementById("getStartedBtn").addEventListener("click", async (e) => {
		e.preventDefault();

		document.getElementById("tab-analysis").classList.add("active");
		document.getElementById("classify").classList.add("show", "active");

		document.getElementById("tab-home").classList.remove("active");
		document.getElementById("home").classList.remove("show", "active");
	});

	document.getElementById("learnMoreBtn").addEventListener("click", (e) => {
  		e.preventDefault();

  		document.getElementById("tab-about").classList.add("active");
		document.getElementById("about").classList.add("show", "active");

		document.getElementById("tab-home").classList.remove("active");
		document.getElementById("home").classList.remove("show", "active");
	});

	 // Button event listeners
	btnIndividual.addEventListener("click", () => {
		resetMode();
		fileForm.classList.remove("d-none");
		fileInput.accept = ".jpg,.jpeg,.png,.bmp,.tiff,.webp";
		btnIndividualContainer.classList.remove("gray-out");
		btnBulkContainer.classList.add("gray-out");
		btnDefaultContainer.classList.add("gray-out");
	});

	btnBulk.addEventListener("click", () => {
		resetMode();
		fileFormBulk.classList.remove("d-none");
		fileInput.accept = ".zip";
		btnBulkContainer.classList.remove("gray-out");
		btnIndividualContainer.classList.add("gray-out");
		btnDefaultContainer.classList.add("gray-out");
	});

	btnDefault.addEventListener("click", () => {
		resetMode();
		defaultWrapper.classList.remove("d-none");
		btnDefaultContainer.classList.remove("gray-out");
		btnIndividualContainer.classList.add("gray-out");
		btnBulkContainer.classList.add("gray-out");
	});

	processDefaultBtn.addEventListener("click", async () => {
		try {
			const response = await fetch("/process/default", {
				method: "POST",
			});

			const result = await response.json();
			if (response.ok) {
				showMessage(result.message || "Default dataset processed.");
			} else {
				showMessage(result.error || "An error occurred.", true);
			}
		} catch (error) {
			showMessage("Failed to process default dataset: " + error.message, true);
		}
	});

	// Handle file form submissions by mode
	handleFileFormSubmit(fileForm, false);
	handleFileFormSubmit(fileFormBulk, true);

});
