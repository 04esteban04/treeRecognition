document.addEventListener("DOMContentLoaded", () => {
	
	// Elements
	const messageBox = document.getElementById("message");

	const defaultWrapper = document.getElementById("processDefaultBtnWrapper");
	const processDefaultBtn = document.getElementById("processDefaultDatasetBtn");

	const resultCardContainer = document.getElementById("resultCardContainer");
	const resultCardGrid = document.getElementById("resultCardGrid");

	const results = document.getElementById("results");
	const toggleViewBtn = document.getElementById("toggleViewBtn");
	const resultTableContainer = document.getElementById("resultTableContainer");
	const resultTableBody = document.getElementById("resultTableBody");
		
	const btnIndividualContainer = document.getElementById("Individual");
	const btnBulkContainer = document.getElementById("In-bulk");
	const btnDefaultContainer = document.getElementById("Default");
	
	const btnIndividual = document.getElementById("button-individual");
	const btnBulk = document.getElementById("button-bulk");
	const btnDefault = document.getElementById("button-default");

	const forms = document.querySelectorAll("form");
	const fileForm = document.getElementById("fileUploadFormIndividual");
	const fileFormBulk = document.getElementById("fileUploadFormBulk");

	const fileUploadFormIndividual = document.querySelector('#fileUploadFormIndividual input[type="file"]');
	const uploadFileBtn = document.getElementById('uploadFileBtn');
	const fileNameSpan = document.getElementById('fileName');

	const bulkFileInput = document.querySelector('#fileUploadFormBulk input[type="file"]');
	const uploadBulkBtn = document.getElementById('uploadBulkBtn');
	const bulkFileNameSpan = document.getElementById('bulkFileName');

	const getStartedBtn = document.getElementById("getStartedBtn");
	const learnMoreBtn = document.getElementById("learnMoreBtn");

	let currentView = "card"; 

	function toggleView() {
		if (currentView === "card") {
			resultCardContainer.classList.add("d-none");
			resultTableContainer.classList.remove("d-none");
			toggleViewBtn.textContent = "Toggle card view";
			currentView = "table";
		} else {
			resultCardContainer.classList.remove("d-none");
			resultTableContainer.classList.add("d-none");
			toggleViewBtn.textContent = "Toggle table view";
			currentView = "card";
		}
	}
	
	// Show processing confirmation message
	function showMessage(msg, isError = false) {
		messageBox.textContent = msg;
		messageBox.classList.remove("d-none", "alert-info", "alert-danger");
		messageBox.classList.add(isError ? "alert-danger" : "alert-info");
	}

	// Reset all modes and hide forms
	function resetMode() {
		fileForm.classList.add("d-none");
		fileFormBulk.classList.add("d-none");
		defaultWrapper.classList.add("d-none");
		messageBox.classList.add("d-none");
		
		results.classList.add("d-none");
	
		// Clear previous values
		resultCardGrid.innerHTML = "";
		resultTableBody.innerHTML = "";
		currentView = "table";
		toggleView();

		btnIndividual.classList.remove("btn-light");
  		btnBulk.classList.remove("btn-light");
		btnDefault.classList.remove("btn-light");
		
		btnIndividual.classList.add("btn-outline-light");
  		btnBulk.classList.add("btn-outline-light");
		btnDefault.classList.add("btn-outline-light");
	}

	// Change loading spinner in buttons when submitting
	function loadingButtons(){

		// Prevent multiple form submits
		forms.forEach(form => {
			form.addEventListener("submit", function (e) {
				const button = form.querySelector(".submit-btn");
				if (button) {
					const textSpan = button.querySelector(".btn-text");
					const spinner = button.querySelector(".spinner-border");
					textSpan.classList.add("d-none");
					spinner.classList.remove("d-none");
					button.disabled = true;
				}
			});
		});

		// Default dataset button
		if (processDefaultBtn) {
			processDefaultBtn.addEventListener("click", function () {
				const textSpan = processDefaultBtn.querySelector(".btn-text");
				const spinner = processDefaultBtn.querySelector(".spinner-border");

				if (textSpan && spinner) {
					textSpan.classList.add("d-none");
					spinner.classList.remove("d-none");
					processDefaultBtn.disabled = true;
				}
			});
		}
	}

	// Reset all buttons with class "submit-btn"
	function resetLoadingButtons() {
		const buttons = document.querySelectorAll(".submit-btn, #processDefaultDatasetBtn");
		buttons.forEach(button => {
			const textSpan = button.querySelector(".btn-text");
			const spinner = button.querySelector(".spinner-border");

			if (textSpan && spinner) {
				textSpan.classList.remove("d-none");
				spinner.classList.add("d-none");
				button.disabled = false;
			}
		});
	}

	// Update button status when data is received
	function showSuccessCheck(button) {
		const textSpan = button.querySelector(".btn-text");
		const spinner = button.querySelector(".spinner-border");
		const checkIcon = button.querySelector(".check-icon");

		if (textSpan && spinner && checkIcon) {
			spinner.classList.add("d-none");
			checkIcon.classList.remove("d-none");
		}

		setTimeout(() => {
			checkIcon.classList.add("d-none");
			button.disabled = false;

			results.scrollIntoView({ behavior: "smooth" });
		}, 1000);
	}

	// Process individual and bulk file upload
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
					
					// Clear previous cards
					resultCardGrid.innerHTML = "";
					resultTableBody.innerHTML = "";

					if (result.results && result.results.length > 0) {
						
						// Show results
						results.classList.remove("d-none");

						result.results.forEach(item => {
							
							// Card view	
							const col = document.createElement("div");
							col.classList.add("col");

							col.innerHTML = `
								<div class="card about-card h-100 shadow-sm">
									<img src="/preprocessed_images/${item["Image name"]}" class="card-img-top mt-2" alt="${item["Image name"]}" style="height: 200px; width: 100%; object-fit: contain;">
									<div class="card-body text-center d-flex flex-column justify-content-center align-items-center">
										<h5 class="card-title mb-2">File: ${item["Image name"]}</h5>
										<hr>
										<table class="table table-bordered table-striped table-dark text-light table-hover">
											<tbody>
												<tr>
													<th scope="row">Real class:</th>
													<td>${item["Real value"] || "N/A"}</td>
												</tr>
												<tr>
													<th scope="row">Predicted class:</th>
													<td>${item["Predicted"]}</td>
												</tr>
												<tr>
													<th scope="row">Accuracy:</th>
													<td>${item["Prob (%)"]}%</td>
												</tr>
												<tr>
													<th scope="row">Prediction correct?</th>
													<td class="${item["Is prediction correct?"] === '✔️' ? 'text-success' : 'text-danger'}">
														${item["Is prediction correct?"]}
													</td>
												</tr>
											</tbody>
										</table>
									</div>
								</div>
							`;
							resultCardGrid.appendChild(col);

							// Table view
							const row = document.createElement("tr");
							row.innerHTML = `
								<td><img src="/preprocessed_images/${item["Image name"]}" alt="${item["Image name"]}" class="img-thumbnail" style="max-width: 100px;"></td>
								<td>${item["Image name"]}</td>
								<td>${item["Real value"]}</td>
								<td>${item["Predicted"]}</td>
								<td>${item["Prob (%)"]}%</td>
								<td class="${item["Is prediction correct?"] === '✔️' ? 'text-success' : 'text-danger'}">
									${item["Is prediction correct?"]}
								</td>
							`;

							resultTableBody.appendChild(row);
						});

						resultCardContainer.classList.remove("d-none");
						
						const button = formElement.querySelector(".submit-btn");
						if (button) {
							showSuccessCheck(button);
						}

					} else {
						showMessage("No results found.");
					}

					resetLoadingButtons(); 

				} else {
					showMessage(result.error || "An error occurred during processing.", true);
				}
			} catch (error) {
				showMessage("Failed to process file: " + error.message, true);
			}
		});
	}

	// Tab navigation from main page buttons
	getStartedBtn.addEventListener("click", async (e) => {
		e.preventDefault();

		document.getElementById("tab-analysis").classList.add("active");
		document.getElementById("classify").classList.add("show", "active");

		document.getElementById("tab-home").classList.remove("active");
		document.getElementById("home").classList.remove("show", "active");
	});

	learnMoreBtn.addEventListener("click", (e) => {
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
		btnIndividualContainer.classList.remove("gray-out");
		btnIndividual.classList.remove("btn-outline-light");
		btnIndividual.classList.add("btn-light");
		btnBulkContainer.classList.add("gray-out");
		btnDefaultContainer.classList.add("gray-out");
	});

	btnBulk.addEventListener("click", () => {
		resetMode();
		fileFormBulk.classList.remove("d-none");
		btnBulkContainer.classList.remove("gray-out");
		btnBulk.classList.remove("btn-outline-light");
		btnBulk.classList.add("btn-light");
		btnIndividualContainer.classList.add("gray-out");
		btnDefaultContainer.classList.add("gray-out");
	});

	btnDefault.addEventListener("click", () => {
		resetMode();
		defaultWrapper.classList.remove("d-none");
		btnDefaultContainer.classList.remove("gray-out");
		btnDefault.classList.remove("btn-outline-light");
		btnDefault.classList.add("btn-light");
		btnIndividualContainer.classList.add("gray-out");
		btnBulkContainer.classList.add("gray-out");
	});

	// Process default dataset
	processDefaultBtn.addEventListener("click", async () => {
		try {
			const response = await fetch("/process/default", {
				method: "POST",
			});

			const result = await response.json();
			if (response.ok) {
				showMessage(result.message || "Default dataset processed.");

				// Clear old cards
				resultCardGrid.innerHTML = "";
				resultTableBody.innerHTML = "";

				if (result.results && result.results.length > 0) {

					// Show results
					results.classList.remove("d-none");

					result.results.forEach(item => {

						// Card view
						const col = document.createElement("div");
						col.classList.add("col");

						col.innerHTML = `
							<div class="card about-card h-100 shadow-sm card-image-wrapper">
								<img src="/preprocessed_images/${item["Image name"]}" class="card-img-top mt-2 card-image" alt="${item["Image name"]}">
								<div class="card-body text-center d-flex flex-column justify-content-center align-items-center">
									<h5 class="card-title mb-2">File: ${item["Image name"]}</h5>
									<hr>
									<table class="table table-bordered table-striped table-dark text-light table-hover">
										<tbody>
											<tr>
												<th scope="row">Real class:</th>
												<td>${item["Real value"]}</td>
											</tr>
											<tr>
												<th scope="row">Predicted class:</th>
												<td>${item["Predicted"]}</td>
											</tr>
											<tr>
												<th scope="row">Accuracy:</th>
												<td>${item["Prob (%)"]}%</td>
											</tr>
											<tr>
												<th scope="row">Prediction correct?</th>
												<td class="${item["Is prediction correct?"] === '✔️' ? 'text-success' : 'text-danger'}">
													${item["Is prediction correct?"]}
												</td>
											</tr>
										</tbody>
									</table>
								</div>
							</div>
						`;

						resultCardGrid.appendChild(col);

						// Table view
						const row = document.createElement("tr");

						row.innerHTML = `
							<td><img src="/preprocessed_images/${item["Image name"]}" alt="${item["Image name"]}" class="img-thumbnail" style="max-width: 100px;"></td>
							<td>${item["Image name"]}</td>
							<td>${item["Real value"]}</td>
							<td>${item["Predicted"]}</td>
							<td>${item["Prob (%)"]}%</td>
							<td class="${item["Is prediction correct?"] === '✔️' ? 'text-success' : 'text-danger'}">
								${item["Is prediction correct?"]}
							</td>
						`;

						resultTableBody.appendChild(row);
					});

					resultCardContainer.classList.remove("d-none");

					showSuccessCheck(processDefaultBtn);

				} else {
					showMessage("No results found.");
				}

				resetLoadingButtons(); 

			} else {
				showMessage(result.error || "An error occurred.", true);
			}
		} catch (error) {
			showMessage("Failed to process default dataset: " + error.message, true);
		}
	});

	// Update file name display when a file is selected
	fileUploadFormIndividual.addEventListener('change', () => {
		if (fileUploadFormIndividual.files.length > 0) {
			fileNameSpan.textContent = fileUploadFormIndividual.files[0].name;
		} else {
			fileNameSpan.textContent = 'No file chosen';
		}
	});

	// Update bulk file name display when a file is selected
	bulkFileInput.addEventListener('change', () => {
		if (bulkFileInput.files.length > 0) {
			bulkFileNameSpan.textContent = bulkFileInput.files[0].name;
		} else {
			bulkFileNameSpan.textContent = 'No file chosen';
		}
	});

	// Toggle between card and table views
	toggleViewBtn.addEventListener("click", toggleView);
	
	// Handle file upload button clicks
	uploadFileBtn.addEventListener('click', () => fileUploadFormIndividual.click());
	uploadBulkBtn.addEventListener('click', () => bulkFileInput.click());

	// Handle file form submissions by mode
	handleFileFormSubmit(fileForm, false);
	handleFileFormSubmit(fileFormBulk, true);

	// Change loading spinner in buttons when submitting
	loadingButtons();

});
