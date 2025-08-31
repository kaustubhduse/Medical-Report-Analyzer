    import { useEffect, useState, useRef } from "react";
    import { useNavigate } from "react-router-dom";
    import { motion } from "framer-motion";
    import Modal from "./Modal"; // Using the enhanced Modal component
    import { FiAlertTriangle } from "react-icons/fi";

    // --- Medical Data Definitions ---
    const medicalEntities = {
    diseases: ["diabetes", "cancer", "asthma", "hypertension", "stroke"],
    medications: ["aspirin", "metformin", "ibuprofen", "insulin", "paracetamol"],
    symptoms: ["fever", "cough", "fatigue", "headache", "dizziness"],
    };

    const fatalDiseases = ["cancer", "stroke", "hypertension"];

    // --- Text Highlighting Function ---
    // Updated with colors that have better contrast on a dark background.
    const highlightEntities = (text) => {
    if (!text) return "";
    let processedText = text;

    Object.entries(medicalEntities).forEach(([category, words]) => {
        words.forEach((word) => {
        const regex = new RegExp(`\\b${word}\\b`, "gi");
        const colorClass =
            category === "diseases"
            ? "text-red-400 font-bold"
            : category === "medications"
            ? "text-blue-400 font-bold"
            : "text-green-400 font-bold"; // Symptoms

        processedText = processedText.replace(
            regex,
            (match) => `<span class="${colorClass}">${match}</span>`
        );
        });
    });

    return processedText.replace(/\n/g, "<br>");
    };

    const Summary = () => {
    const [summary, setSummary] = useState("");
    const [loading, setLoading] = useState(true);
    const [isAlertModalOpen, setIsAlertModalOpen] = useState(false);
    const navigate = useNavigate();
    // Use a ref to ensure the alert is shown only once per summary load
    const alertShownRef = useRef(false);

    // --- Fetch the summary text file ---
    useEffect(() => {
        fetch("/summary.txt")
        .then((response) => response.text())
        .then((text) => setSummary(text))
        .catch((error) => console.error("Error loading summary:", error))
        .finally(() => setLoading(false));
    }, []);

    // --- Check for fatal diseases and trigger the modal ---
    useEffect(() => {
        if (!summary || alertShownRef.current) return;

        const hasFatalDisease = fatalDiseases.some((disease) =>
        summary.toLowerCase().includes(disease)
        );

        if (hasFatalDisease) {
        setIsAlertModalOpen(true);
        alertShownRef.current = true; // Mark alert as shown
        }
    }, [summary]);

    const handleSearchHospitals = () => {
        setIsAlertModalOpen(false);
        navigate("/search");
    };

    return (
        <div className="relative min-h-screen flex flex-col items-center bg-slate-900 p-6 overflow-hidden">
        {/* Animated background shapes */}
        <div className="absolute top-0 -left-4 w-72 h-72 bg-blue-500 rounded-full mix-blend-lighten filter blur-xl opacity-70 animate-blob"></div>
        <div className="absolute top-0 -right-4 w-72 h-72 bg-purple-500 rounded-full mix-blend-lighten filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>

        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="relative w-full max-w-4xl p-8 bg-white/5 backdrop-blur-md border border-white/20 shadow-2xl rounded-3xl z-10"
        >
            <h1 className="text-3xl font-bold mb-4 text-white">
            Medical Report Summary
            </h1>

            {/* --- Color Legend --- */}
            <div className="flex flex-wrap gap-x-6 gap-y-2 mb-6 text-sm">
            <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-red-400"></div>
                <span className="text-gray-300">Diseases</span>
            </div>
            <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-blue-400"></div>
                <span className="text-gray-300">Medications</span>
            </div>
            <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-green-400"></div>
                <span className="text-gray-300">Symptoms</span>
            </div>
            </div>

            <div className="bg-slate-900/50 p-6 rounded-lg min-h-[400px]">
            {loading ? (
                <div className="flex justify-center items-center h-full">
                <div className="w-12 h-12 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                </div>
            ) : (
                <p
                className="whitespace-pre-wrap break-words text-gray-200 text-lg leading-relaxed"
                dangerouslySetInnerHTML={{ __html: highlightEntities(summary) }}
                ></p>
            )}
            </div>
        </motion.div>

        {/* --- Custom Alert Modal --- */}
        <Modal
            isOpen={isAlertModalOpen}
            onClose={() => setIsAlertModalOpen(false)}
            title="Health Alert"
        >
            <div className="text-center">
            <FiAlertTriangle className="mx-auto text-yellow-400 text-5xl mb-4" />
            <p className="text-gray-300 mb-6">
                A potentially serious condition has been detected in the summary.
                Would you like to find a nearby hospital?
            </p>
            <div className="flex justify-center gap-4">
                <button
                onClick={() => setIsAlertModalOpen(false)}
                className="px-6 py-2 bg-slate-600 text-white rounded-md hover:bg-slate-500 transition-colors"
                >
                Continue Reading
                </button>
                <button
                onClick={handleSearchHospitals}
                className="px-6 py-2 bg-yellow-500 text-white rounded-md hover:bg-yellow-600 transition-colors"
                >
                Search Hospitals
                </button>
            </div>
            </div>
        </Modal>
        </div>
    );
    };

    export default Summary;
