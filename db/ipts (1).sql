-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 11, 2023 at 09:46 PM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.0.28

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `ipts`
--

-- --------------------------------------------------------

--
-- Table structure for table `attendance`
--

CREATE TABLE `attendance` (
  `id` int(30) NOT NULL,
  `att_id` varchar(30) NOT NULL,
  `dept_tid` varchar(30) NOT NULL,
  `cl_id` varchar(30) NOT NULL,
  `year` int(30) NOT NULL,
  `ad_no` varchar(30) NOT NULL,
  `att_status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `attendance`
--

INSERT INTO `attendance` (`id`, `att_id`, `dept_tid`, `cl_id`, `year`, `ad_no`, `att_status`) VALUES
(0, '19f53486-e5c7-4627-a774-6b0957', '2', 'ce04', 2, 'l21ds006', 1),
(0, '19f53486-e5c7-4627-a774-6b0957', '2', 'ce04', 2, 'l21ds008', 1),
(0, '19f53486-e5c7-4627-a774-6b0957', '2', 'ce04', 2, 'l21ds009', 1),
(0, '19f53486-e5c7-4627-a774-6b0957', '2', 'ce04', 2, 'l21ds007', 1),
(0, 'f1500567-d725-4c93-90be-73017d', '2', 'ce04', 2, 'l21ds006', 1),
(0, 'f1500567-d725-4c93-90be-73017d', '2', 'ce04', 2, 'l21ds008', 1),
(0, 'f1500567-d725-4c93-90be-73017d', '2', 'ce04', 2, 'l21ds009', 1),
(0, 'f1500567-d725-4c93-90be-73017d', '2', 'ce04', 2, 'l21ds007', 1);

-- --------------------------------------------------------

--
-- Table structure for table `attendance_record`
--

CREATE TABLE `attendance_record` (
  `att_id` varchar(30) NOT NULL,
  `date` date NOT NULL,
  `year` int(30) NOT NULL,
  `dept_id` varchar(30) NOT NULL,
  `cl_id` varchar(30) NOT NULL,
  `teacher_id` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `attendance_record`
--

INSERT INTO `attendance_record` (`att_id`, `date`, `year`, `dept_id`, `cl_id`, `teacher_id`) VALUES
('517acbd6-ee3c-4237-b35e-d2997c', '2023-06-12', 2, '2', 'ce04', '1'),
('edaad326-1d73-42a4-b136-a7ecf5', '2023-06-12', 2, '2', 'ce04', '1'),
('19f53486-e5c7-4627-a774-6b0957', '2023-06-12', 2, '2', 'ce04', '1'),
('f1500567-d725-4c93-90be-73017d', '2023-06-12', 2, '2', 'ce04', '1');

-- --------------------------------------------------------

--
-- Table structure for table `department`
--

CREATE TABLE `department` (
  `dept_id` varchar(30) NOT NULL,
  `name` varchar(30) NOT NULL,
  `cl_id` varchar(30) NOT NULL,
  `className` varchar(30) NOT NULL,
  `year` int(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `department`
--

INSERT INTO `department` (`dept_id`, `name`, `cl_id`, `className`, `year`) VALUES
('1', 'CS', 'cs01', 'CS CSE', 1),
('1', 'CS', 'cs02', 'CS DS', 1),
('1', 'CS', 'cs03', 'CS AIML', 1),
('1', 'CS', 'cs04', 'CS CSE', 2),
('1', 'CS', 'cs05', 'CS DS', 2),
('1', 'CS', 'cs06', 'CS CSE', 3),
('1', 'CS', 'cs07', 'CS DS', 3),
('1', 'CS', 'cs08', 'CS CSE', 4),
('2', 'CE', 'ce01', 'CE C-A', 1),
('2', 'CE', 'ce02', 'CE C-B', 1),
('2', 'CE', 'ce03', 'CE C-A', 2),
('2', 'CE', 'ce04', 'CE C-B', 2),
('2', 'CE', 'ce05', 'CE C-A', 3),
('2', 'CE', 'ce06', 'CE C-B', 3),
('2', 'CE', 'ce07', 'CE C-A', 4),
('2', 'CE', 'ce08', 'CE C-B', 4),
('3', 'ME', 'me01', 'ME M1', 1),
('3', 'ME', 'me02', 'ME M2', 1),
('3', 'ME', 'me03', 'ME M1', 2),
('3', 'ME', 'me04', 'ME M2', 2),
('3', 'ME', 'me05', 'ME M1', 3),
('3', 'ME', 'me06', 'ME M2', 3),
('3', 'ME', 'me07', 'ME M1', 4),
('3', 'ME', 'me08', 'ME M2', 4),
('4', 'EEE', 'eee01', 'EEE E1', 1),
('4', 'EEE', 'eee02', 'EEE E2', 1),
('4', 'EEE', 'eee03', 'EEE E3', 1),
('4', 'EEE', 'eee04', 'EEE E1', 2),
('4', 'EEE', 'eee05', 'EEE E2', 2),
('4', 'EEE', 'eee06', 'EEE E3', 2),
('4', 'EEE', 'eee07', 'EEE E1', 3),
('4', 'EEE', 'eee08', 'EEE E2', 3),
('4', 'EEE', 'eee09', 'EEE E3', 3),
('4', 'EEE', 'eee10', 'EEE E1', 4),
('4', 'EEE', 'eee11', 'EEE E2', 4),
('4', 'EEE', 'eee12', 'EEE E3', 4),
('5', 'EC', 'ec01', 'EC EC1', 1),
('5', 'EC', 'ec02', 'EC EC2', 1),
('5', 'EC', 'ec03', 'EC EC1', 2),
('5', 'EC', 'ec04', 'EC EC2', 2),
('5', 'EC', 'ec05', 'EC EC1', 3),
('5', 'EC', 'ec06', 'EC EC2', 3),
('5', 'EC', 'ec07', 'EC EC1', 4),
('5', 'EC', 'ec08', 'EC EC2', 4),
('5', 'CE', 'ce01', 'CE C1', 1),
('5', 'CE', 'ce02', 'CE C2', 1),
('5', 'CE', 'ce03', 'CE C1', 2),
('5', 'CE', 'ce04', 'CE C2', 2),
('5', 'CE', 'ce05', 'CE C1', 3),
('5', 'CE', 'ce06', 'CE C2', 3),
('5', 'CE', 'ce07', 'CE C1', 4),
('5', 'CE', 'ce08', 'CE C2', 4);

-- --------------------------------------------------------

--
-- Table structure for table `students`
--

CREATE TABLE `students` (
  `StudentID` int(20) NOT NULL,
  `ad_no` varchar(20) DEFAULT NULL,
  `cl_id` varchar(11) DEFAULT NULL,
  `name` varchar(100) DEFAULT NULL,
  `Age` int(11) DEFAULT NULL,
  `Gender` varchar(10) DEFAULT NULL,
  `dept_id` varchar(50) DEFAULT NULL,
  `Year` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `students`
--

INSERT INTO `students` (`StudentID`, `ad_no`, `cl_id`, `name`, `Age`, `Gender`, `dept_id`, `Year`) VALUES
(1, 'l21ds008', 'ce04', 'mohammed rinshad p', 23, 'male', '2', 2),
(2, 'l21ds007', 'ce04', 'Muralikrishna K S', 22, 'male', '2', 2),
(3, 'l21ds009', 'ce04', 'Ashiq joju CJ', 22, 'male', '2', 2),
(4, 'l21ds006', 'ce04', 'ARAVIND.R', 23, 'male', '2', 2);

-- --------------------------------------------------------

--
-- Table structure for table `teachers`
--

CREATE TABLE `teachers` (
  `teacher_id` int(30) NOT NULL,
  `name` varchar(30) NOT NULL,
  `username` varchar(30) NOT NULL,
  `password` varchar(30) NOT NULL,
  `major` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `teachers`
--

INSERT INTO `teachers` (`teacher_id`, `name`, `username`, `password`, `major`) VALUES
(1, 'John', 'teach1', '1234', 'Mathematics'),
(2, 'Teacher 2', 'teacher2', 'password2', 'Science'),
(3, 'Teacher 3', 'teacher3', 'password3', 'English'),
(4, 'Teacher 4', 'teacher4', 'password4', 'History'),
(5, 'Teacher 5', 'teacher5', 'password5', 'Geography');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `students`
--
ALTER TABLE `students`
  ADD PRIMARY KEY (`StudentID`);

--
-- Indexes for table `teachers`
--
ALTER TABLE `teachers`
  ADD PRIMARY KEY (`teacher_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `students`
--
ALTER TABLE `students`
  MODIFY `StudentID` int(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1021;

--
-- AUTO_INCREMENT for table `teachers`
--
ALTER TABLE `teachers`
  MODIFY `teacher_id` int(30) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
