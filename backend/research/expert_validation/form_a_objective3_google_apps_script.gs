/*
 * Objective 3 Form A: independent expert/traditional-cutter recommendation.
 *
 * Use this Google Apps Script to create or intentionally update the blind
 * Form A instrument. Experts must complete this form before seeing any system
 * result. The form contains no system yield, shape, orientation, optimizer
 * result, AI prediction, or diagnostic output.
 *
 * Default safety behavior:
 * - Leave EXISTING_FORM_ID blank to create a new final Form A.
 * - Set EXISTING_FORM_ID only if you intentionally want to replace the items
 *   in an existing Form A while preserving its response destination.
 * - The normalized response sheet uses the exact CSV schema expected by
 *   backend/research/expert_validation.
 */

const FORM_A_VERSION = "objective3_external_expert_form_a_v1";
const FORM_TITLE = "Objective 3 Form A - Independent Gem-Cutting Recommendation";
const EXISTING_FORM_ID = "";
const EXISTING_RESPONSE_SPREADSHEET_ID = "";
const NORMALIZED_SHEET_NAME = "Form A Normalized Responses";

const FINAL_SPECIMENS = Object.freeze([
  "QZ-01",
  "QZ-03",
  "QZ-05",
  "QZ-08",
  "QZ-14",
]);

const REMOVED_SPECIMENS = Object.freeze([
  "QZ-09",
  "QZ-30",
]);

const SCHEMA_HEADERS = Object.freeze([
  "response_id",
  "submitted_at",
  "expert_id",
  "expert_name_or_code",
  "years_experience",
  "consent_independent_blind",
  "form_version",
  "specimen_id",
  "recommended_cut_shape",
  "gem_count",
  "retained_weight_ct",
  "orientation_description",
  "orientation_vector_x",
  "orientation_vector_y",
  "orientation_vector_z",
  "defects_or_areas_to_avoid",
  "rationale",
  "confidence_1_to_5",
  "manufacturable_with_standard_saw",
  "notes",
]);

const IDENTITY_TITLES = Object.freeze({
  expertId: "Expert ID",
  expertName: "Expert name or code",
  yearsExperience: "Years of gem-cutting experience",
  consent: "Independent blind recommendation consent",
});

const SPECIMEN_FIELD_TITLES = Object.freeze({
  recommendedCutShape: "Recommended cut shape",
  gemCount: "Recommended number of finished gems",
  retainedWeightCt: "Estimated total retained weight (ct)",
  orientationDescription: "Recommended orientation",
  orientationVectorX: "Optional orientation vector x",
  orientationVectorY: "Optional orientation vector y",
  orientationVectorZ: "Optional orientation vector z",
  visibleDefects: "Visible defects / inclusions considered",
  areasToAvoid: "Areas to avoid during cutting",
  rationale: "Reasoning / cutting rationale",
  confidence: "Confidence rating",
  manufacturable: "Manufacturable with standard saw/workshop methods",
  notes: "Optional comments",
});

function createFinalObjective3FormA() {
  validateFinalSpecimens_();

  const form = getOrCreateForm_();
  clearFormItems_(form);
  form.setTitle(FORM_TITLE);
  form.setDescription([
    "Form A is an independent expert recommendation instrument.",
    "Complete all recommendations before reviewing any system or optimizer result.",
    "Do not use AI predictions, system yield, system shape, system orientation, or optimizer output.",
    "Use only the raw specimen evidence supplied separately for the final specimen set.",
    "Final specimens: " + FINAL_SPECIMENS.join(", ") + ".",
  ].join("\n"));
  form.setCollectEmail(false);
  form.setAllowResponseEdits(false);
  form.setLimitOneResponsePerUser(false);
  form.setProgressBar(true);
  form.setShowLinkToRespondAgain(false);
  form.setConfirmationMessage(
    "Thank you. Do not review system results until Form A responses are frozen."
  );

  addIdentitySection_(form);
  FINAL_SPECIMENS.forEach(function(specimenId) {
    addSpecimenSection_(form, specimenId);
  });
  validateFormDoesNotContainRemovedSpecimens_(form);

  const spreadsheet = getOrCreateResponseSpreadsheet_();
  form.setDestination(FormApp.DestinationType.SPREADSHEET, spreadsheet.getId());
  ensureNormalizedSheet_(spreadsheet);

  PropertiesService.getScriptProperties().setProperty("FORM_A_ID", form.getId());
  PropertiesService.getScriptProperties().setProperty(
    "FORM_A_RESPONSE_SPREADSHEET_ID",
    spreadsheet.getId()
  );

  Logger.log("Form edit URL: " + form.getEditUrl());
  Logger.log("Form live URL: " + form.getPublishedUrl());
  Logger.log("Response spreadsheet URL: " + spreadsheet.getUrl());
  return {
    formId: form.getId(),
    editUrl: form.getEditUrl(),
    liveUrl: form.getPublishedUrl(),
    spreadsheetId: spreadsheet.getId(),
    spreadsheetUrl: spreadsheet.getUrl(),
  };
}

function installNormalizeFormATrigger() {
  const formId = getStoredFormId_();
  const form = FormApp.openById(formId);
  ScriptApp.getProjectTriggers().forEach(function(trigger) {
    if (trigger.getHandlerFunction() === "normalizeFormAResponse") {
      ScriptApp.deleteTrigger(trigger);
    }
  });
  ScriptApp.newTrigger("normalizeFormAResponse")
    .forForm(form)
    .onFormSubmit()
    .create();
  Logger.log("Installed normalizeFormAResponse trigger for form " + formId);
}

function normalizeFormAResponse(event) {
  if (!event || !event.response) {
    throw new Error("normalizeFormAResponse requires a Form submit event.");
  }
  const response = event.response;
  const answers = answersByTitle_(response);
  const spreadsheet = SpreadsheetApp.openById(getStoredSpreadsheetId_());
  const sheet = ensureNormalizedSheet_(spreadsheet);

  const responseId = response.getId ? response.getId() : Utilities.getUuid();
  const submittedAt = response.getTimestamp().toISOString();
  const expertId = answer_(answers, IDENTITY_TITLES.expertId);
  const expertName = answer_(answers, IDENTITY_TITLES.expertName);
  const yearsExperience = answer_(answers, IDENTITY_TITLES.yearsExperience);
  const consent = normalizeConsent_(answer_(answers, IDENTITY_TITLES.consent));

  const rows = FINAL_SPECIMENS.map(function(specimenId) {
    const visibleDefects = answer_(
      answers,
      specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.visibleDefects)
    );
    const areasToAvoid = answer_(
      answers,
      specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.areasToAvoid)
    );
    return [
      responseId,
      submittedAt,
      expertId,
      expertName,
      yearsExperience,
      consent,
      FORM_A_VERSION,
      specimenId,
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.recommendedCutShape)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.gemCount)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.retainedWeightCt)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.orientationDescription)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.orientationVectorX)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.orientationVectorY)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.orientationVectorZ)),
      [
        "Visible defects / inclusions considered: " + visibleDefects,
        "Areas to avoid during cutting: " + areasToAvoid,
      ].join(" | "),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.rationale)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.confidence)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.manufacturable)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.notes)),
    ];
  });

  sheet.getRange(sheet.getLastRow() + 1, 1, rows.length, SCHEMA_HEADERS.length)
    .setValues(rows);
}

function addIdentitySection_(form) {
  form.addSectionHeaderItem()
    .setTitle("Expert identity and independence")
    .setHelpText("These fields support traceability without showing any system result.");
  form.addTextItem()
    .setTitle(IDENTITY_TITLES.expertId)
    .setHelpText("Use the assigned expert ID/code if available.")
    .setRequired(true);
  form.addTextItem()
    .setTitle(IDENTITY_TITLES.expertName)
    .setHelpText("Name or role-coded identifier.")
    .setRequired(true);
  form.addTextItem()
    .setTitle(IDENTITY_TITLES.yearsExperience)
    .setHelpText("Enter a number, for example 8 or 12.5.")
    .setRequired(true);
  form.addMultipleChoiceItem()
    .setTitle(IDENTITY_TITLES.consent)
    .setHelpText(
      "Confirm this recommendation is independent and completed before seeing system results."
    )
    .setChoiceValues(["YES", "NO"])
    .setRequired(true);
}

function addSpecimenSection_(form, specimenId) {
  form.addPageBreakItem()
    .setTitle(specimenId)
    .setHelpText(
      "Use only the raw specimen evidence supplied separately. Do not consult any system result."
    );
  form.addTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.recommendedCutShape))
    .setHelpText("Recommended finished cut shape or shape mix.")
    .setRequired(true);
  form.addTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.gemCount))
    .setHelpText("Integer number of finished gems. Use 0 only if no feasible cut is recommended.")
    .setRequired(true);
  form.addTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.retainedWeightCt))
    .setHelpText("Estimated total retained finished weight in carats.")
    .setRequired(true);
  form.addParagraphTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.orientationDescription))
    .setHelpText("Describe table/crown/axis orientation using the supplied raw specimen evidence.")
    .setRequired(true);
  form.addTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.orientationVectorX))
    .setHelpText("Optional numeric vector component if you use a coordinate convention.")
    .setRequired(false);
  form.addTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.orientationVectorY))
    .setHelpText("Optional numeric vector component if you use a coordinate convention.")
    .setRequired(false);
  form.addTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.orientationVectorZ))
    .setHelpText("Optional numeric vector component if you use a coordinate convention.")
    .setRequired(false);
  form.addParagraphTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.visibleDefects))
    .setHelpText("Visible defects, inclusions, fractures, clouds, or none observed.")
    .setRequired(true);
  form.addParagraphTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.areasToAvoid))
    .setHelpText("Areas to avoid while cutting, or none.")
    .setRequired(true);
  form.addParagraphTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.rationale))
    .setHelpText("Explain the cutting rationale without referencing system output.")
    .setRequired(true);
  form.addScaleItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.confidence))
    .setBounds(1, 5)
    .setLabels("Low confidence", "High confidence")
    .setRequired(true);
  form.addMultipleChoiceItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.manufacturable))
    .setChoiceValues(["YES", "NO", "UNCERTAIN"])
    .setRequired(true);
  form.addParagraphTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.notes))
    .setHelpText("Optional comments.")
    .setRequired(false);
}

function getOrCreateForm_() {
  if (EXISTING_FORM_ID) {
    return FormApp.openById(EXISTING_FORM_ID);
  }
  return FormApp.create(FORM_TITLE);
}

function getOrCreateResponseSpreadsheet_() {
  if (EXISTING_RESPONSE_SPREADSHEET_ID) {
    return SpreadsheetApp.openById(EXISTING_RESPONSE_SPREADSHEET_ID);
  }
  return SpreadsheetApp.create("Objective 3 Form A Responses");
}

function clearFormItems_(form) {
  const items = form.getItems();
  for (let index = items.length - 1; index >= 0; index -= 1) {
    form.deleteItem(items[index]);
  }
}

function ensureNormalizedSheet_(spreadsheet) {
  let sheet = spreadsheet.getSheetByName(NORMALIZED_SHEET_NAME);
  if (!sheet) {
    sheet = spreadsheet.insertSheet(NORMALIZED_SHEET_NAME);
  }
  const existingHeader = sheet.getRange(1, 1, 1, SCHEMA_HEADERS.length).getValues()[0];
  const hasHeader = existingHeader.some(function(value) {
    return value !== "";
  });
  if (!hasHeader) {
    sheet.getRange(1, 1, 1, SCHEMA_HEADERS.length).setValues([SCHEMA_HEADERS]);
  }
  const header = sheet.getRange(1, 1, 1, SCHEMA_HEADERS.length).getValues()[0];
  if (header.join("\t") !== SCHEMA_HEADERS.join("\t")) {
    throw new Error(
      "Normalized sheet header does not match backend expert-validation schema."
    );
  }
  return sheet;
}

function answersByTitle_(response) {
  const answers = {};
  response.getItemResponses().forEach(function(itemResponse) {
    answers[itemResponse.getItem().getTitle()] = stringifyAnswer_(
      itemResponse.getResponse()
    );
  });
  return answers;
}

function answer_(answers, title) {
  return answers[title] || "";
}

function stringifyAnswer_(value) {
  if (Array.isArray(value)) {
    return value.join("; ");
  }
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function normalizeConsent_(value) {
  return String(value || "").trim().toUpperCase() === "YES" ? "YES" : "NO";
}

function specimenTitle_(specimenId, fieldTitle) {
  return specimenId + " - " + fieldTitle;
}

function getStoredFormId_() {
  const formId = EXISTING_FORM_ID ||
    PropertiesService.getScriptProperties().getProperty("FORM_A_ID");
  if (!formId) {
    throw new Error("Run createFinalObjective3FormA first.");
  }
  return formId;
}

function getStoredSpreadsheetId_() {
  const spreadsheetId = EXISTING_RESPONSE_SPREADSHEET_ID ||
    PropertiesService.getScriptProperties().getProperty("FORM_A_RESPONSE_SPREADSHEET_ID");
  if (!spreadsheetId) {
    throw new Error("Run createFinalObjective3FormA first.");
  }
  return spreadsheetId;
}

function validateFinalSpecimens_() {
  REMOVED_SPECIMENS.forEach(function(specimenId) {
    if (FINAL_SPECIMENS.indexOf(specimenId) !== -1) {
      throw new Error("Removed specimen appears in final Form A set: " + specimenId);
    }
  });
  if (FINAL_SPECIMENS.length !== 5) {
    throw new Error("Final Form A specimen set must contain exactly five specimens.");
  }
}

function validateFormDoesNotContainRemovedSpecimens_(form) {
  const text = form.getItems().map(function(item) {
    return item.getTitle() + " " + item.getHelpText();
  }).join("\n");
  REMOVED_SPECIMENS.forEach(function(specimenId) {
    if (text.indexOf(specimenId) !== -1) {
      throw new Error("Form still contains removed specimen: " + specimenId);
    }
  });
}
